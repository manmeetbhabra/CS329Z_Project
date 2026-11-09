"""Score a system's reports against the answer key.

Usage:
    uv run python -m bench.score thacker_pass_clean runs/<run_name>
    uv run python -m bench.score thacker_pass_clean --oracle      # self-check: should score 1.0

Input: <runs_dir>/<case_id>.json for each case, in the format of bench/schema/report.schema.json.
A missing report file counts as an empty report.

Only plans with no known gaps can be scored (thacker_pass_clean). main() asserts this.

The rule (one rule, see hit()):
  A flag catches the planted flaw if it names the right checklist item (any one of the case's
  checklist_ids) at the right place (any one of the case's locations).
  * Right place = same file and, if both name a section, the same section or a subsection of it;
    otherwise pages within +/-1. A flag with no section and no pages has no place, and never matches.
  * One exception: when the flaw deleted a whole document, there is nothing left to point at, so
    the right checklist item alone is enough.
  * The flag's type is not used.
  Each case has one planted flaw. The first flag that catches it counts; later ones are
  duplicates (neither right nor wrong). Every other flag is a false alarm.

Metrics
  recall       caught / planted flaws                       (overall, by type, by difficulty)
  precision    caught / (flags - duplicates)
  false alarms on the clean control case, and per case on average.

How the pieces fit (top to bottom):
  main()          reads the answer key (refusing one with known gaps) and the reports, then
                  calls score() and to_markdown()
  score()         runs score_case() on every case
  score_case()    turns the JSON into a Case and Flags, asks hit() about each flag, and counts
  hit()           right checklist item at the right place?
  overlaps(), same_place(), _same_section()   the location tests underneath hit()
  aggregate()     turns the per-case rows into recall, precision and false-alarm counts

The three types (see their docstrings):
  Location   one spot in the plan: a file, maybe a section, maybe some pages
  Flag       one finding the agent returned: a checklist item it says is violated, and where
  Case       one test case: a copy of the plan with one planted flaw (or none, for the control)

The examples in the docstrings below are doctests; check them with
    uv run python -m doctest bench/score.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the project folder (bench/ is inside it)
PAGE_TOLERANCE = 1  # page numbers within this many pages of each other count as the same place


# ---------------------------------------------------------------------------------------------
# The three types. JSON dicts are turned into these once, as they come in (see score_case).
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Location:
    """One spot in the plan: a file, maybe a section, maybe some page numbers. Nothing else.

    >>> Location("body.md", "3.5", (51,))
    Location(file='body.md', section='3.5', pages=(51,))
    >>> Location("appendix_E.md")          # a whole file
    Location(file='appendix_E.md', section=None, pages=())
    """
    file: str | None              # "body.md", "appendix_E.md", ...
    section: str | None = None    # "3.5", "6.4.8", ...
    pages: tuple[int, ...] = ()   # always a tuple; a case's single "page" becomes (51,)


@dataclass(frozen=True)
class Flag:
    """One finding the agent returned: a checklist item it says is violated, and where.

    A flag has one location, or none: the report schema gives it a single file and section
    (with a list of pages). If the agent saw a problem in two places it writes two flags.
    `type` is kept for reports and analysis; the scoring rule doesn't use it.
    """
    checklist_id: str | None
    type: str | None                         # missing | vague | inconsistent | implausible
    location: Location | None = None         # None if the agent gave no file/section/pages
    raw: dict = field(default_factory=dict, compare=False, repr=False)  # original, for output

    @classmethod
    def from_dict(cls, d: dict) -> Flag:
        """Build a Flag from one entry of a report's "flags" list.

        The location fields sit on the flag itself; they become one Location, or None if the
        flag gives no file, section or pages.

        >>> Flag.from_dict({"checklist_id": "ops-designs", "type": "inconsistent",
        ...                 "file": "body.md", "section": "3.5", "pages": [51], "explanation": "x"})
        Flag(checklist_id='ops-designs', type='inconsistent', location=Location(file='body.md', section='3.5', pages=(51,)))
        >>> Flag.from_dict({"checklist_id": "ops-designs", "type": "missing", "file": None}).location is None
        True
        """
        location = None
        if d.get("file") or d.get("section") or d.get("pages"):
            location = Location(file=d.get("file") or None,
                                section=str(d["section"]) if d.get("section") else None,
                                pages=tuple(d.get("pages") or ()))
        return cls(checklist_id=d.get("checklist_id"), type=d.get("type"), location=location, raw=d)


@dataclass(frozen=True)
class Case:
    """One test case: a copy of the plan with one planted flaw (type "none" for the control).

    A flag catches the flaw with ONE of the checklist_ids at ONE of the locations. The lists
    mean "any one of these is a correct answer": a flaw can fairly be described by more than
    one checklist item, and can show up in more than one spot (e.g. both sides of an
    inconsistency). The items and the spots are not paired.
    """
    id: str                                  # "TP-021"
    type: str                                # missing | vague | inconsistent | implausible | none
    difficulty: str                          # easy | medium | subtle | none
    checklist_ids: tuple[str, ...]           # any one is a correct item
    locations: tuple[Location, ...]          # any one is a correct spot
    removes_a_file: bool = False             # the flaw deleted a whole document

    @classmethod
    def from_dict(cls, d: dict) -> Case:
        """Build a Case from one entry of key["cases"].

        Each answer-key location has a single "page" (or None); it becomes a pages tuple. A
        location with role "file" means the planted flaw deleted that whole document.

        >>> c = Case.from_dict({"case_id": "TP-001", "type": "missing", "difficulty": "easy",
        ...     "checklist_ids": ["ops-spill-contingency"],
        ...     "locations": [{"file": "appendix_E.md", "section": None, "page": None, "role": "file"}]})
        >>> c.locations, c.removes_a_file
        ((Location(file='appendix_E.md', section=None, pages=()),), True)
        >>> c = Case.from_dict({"case_id": "TP-021", "type": "inconsistent", "difficulty": "easy",
        ...     "checklist_ids": ["ops-designs"],
        ...     "locations": [{"file": "body.md", "section": "3.5", "page": 51, "role": "content"},
        ...                   {"file": "body.md", "section": "3.5", "page": 51, "role": "related"}]})
        >>> c.locations[0], c.removes_a_file
        (Location(file='body.md', section='3.5', pages=(51,)), False)
        """
        locations = tuple(
            Location(file=l.get("file") or None,
                     section=str(l["section"]) if l.get("section") else None,
                     pages=(l["page"],) if l.get("page") is not None else ())
            for l in d.get("locations") or [])
        return cls(id=d["case_id"], type=d["type"], difficulty=d["difficulty"],
                   checklist_ids=tuple(d.get("checklist_ids") or ()),
                   locations=locations,
                   removes_a_file=any(l.get("role") == "file" for l in d.get("locations") or []))


# ---------------------------------------------------------------------------------------------
# Location tests: does a flag point at the same place as the planted flaw?
# ---------------------------------------------------------------------------------------------

def _same_section(a: str, b: str) -> bool:
    """True if two section numbers are the same section, or one is a subsection of the other.

    >>> _same_section("6.4", "6.4.8"), _same_section("6.4.8", "6.4")
    (True, True)
    >>> _same_section("3.4", "3.40")       # a different section, not a subsection
    False
    >>> _same_section("3.4.1", "3.4.2")    # siblings do not match
    False
    """
    # The "+ '.'" stops 3.4 from matching 3.40.
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def same_place(a: Location, b: Location) -> bool:
    """True if two spots are the same place.

      1. Both name a file and the files differ: no.
      2. Both name a section: the sections decide (a subsection counts).
      3. Otherwise: some pages within PAGE_TOLERANCE (1) of each other.
    A spot with no section and no pages (just a file) has nothing to compare, so it never matches.

    >>> spot = Location("body.md", "6.4.8", (111,))
    >>> same_place(Location("body.md", "6.4"), spot)            # the parent section
    True
    >>> same_place(Location("body.md", "6.3", (111,)), spot)    # sections decide, not pages
    False
    >>> same_place(Location("body.md", None, (112,)), spot)     # no section on one side: pages
    True
    >>> same_place(Location("appendix_C.md", "6.4.8"), spot)    # different files
    False
    >>> same_place(Location("body.md"), spot)                   # just a file: not a place
    False
    """
    # 1. Different files never match.
    if a.file and b.file and a.file != b.file:
        return False
    # 2. If both name a section, the sections decide.
    if a.section and b.section:
        return _same_section(a.section, b.section)
    # 3. Otherwise compare pages, allowing +/- PAGE_TOLERANCE.
    return any(abs(pa - pb) <= PAGE_TOLERANCE for pa in a.pages for pb in b.pages)


def overlaps(flag: Flag, case: Case) -> bool:
    """Does the flag's spot match any of the case's spots? False if the flag has no location."""
    return flag.location is not None and any(same_place(flag.location, l) for l in case.locations)


# ---------------------------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------------------------

def hit(flag: Flag, case: Case) -> bool:
    """Did this flag find the flaw planted in this case? Right checklist item at the right place.

      1. Is flag.checklist_id one of case.checklist_ids?   no  -> False
      2. Did the flaw delete a whole document?             yes -> True (nothing to point at)
      3. Is the flag's spot one of the case's spots?       -> that answer

    Example, modelled on TP-031: the pond fences are made 2 ft tall (an implausible flaw,
    item pf-exclusion) in three places.
    >>> case = Case("TP-031", "implausible", "easy", ("pf-exclusion",),
    ...             (Location("body.md", "3.17", (72,)), Location("body.md", "3.24.6", (83,)),
    ...              Location("body.md", "4.6", (92,))))
    >>> def flag(item, section=None, file="body.md", ftype="implausible"):
    ...     return Flag(item, ftype, Location(file, section) if file else None)
    >>> hit(flag("pf-exclusion", "3.17"), case)                         # right item, right place
    True
    >>> hit(flag("pf-exclusion", "4.6"), case)                          # any of the spots will do
    True
    >>> hit(flag("pf-exclusion", "3.17", ftype="inconsistent"), case)   # type isn't scored
    True
    >>> hit(flag("pf-exclusion", "9.9"), case)                          # right item, wrong place
    False
    >>> hit(flag("pf-exclusion", file=None), case)                      # no location
    False
    >>> hit(flag("pf-exclusion"), case)                                 # a file but not where in it
    False
    >>> hit(flag("pf-wildlife-tes", "3.17"), case)                      # wrong item
    False

    A deleted document (TP-001 leaves out appendix_E.md): the agent can only cite the lines that
    refer to it, so the right item alone is enough.
    >>> removed = Case("TP-001", "missing", "easy", ("ops-spill-contingency",),
    ...                (Location("appendix_E.md"),), removes_a_file=True)
    >>> hit(flag("ops-spill-contingency", "4.12"), removed)             # cites body §4.12
    True
    >>> hit(flag("ops-spill-contingency", file=None), removed)          # no location
    True
    >>> hit(flag("pf-fire", "4.12"), removed)                           # wrong item
    False

    The clean control has no checklist items, so nothing ever hits it.
    >>> control = Case("TP-000", "none", "none", (), ())
    >>> hit(flag("pf-exclusion", "3.17"), control)
    False
    """
    # 1. Wrong item (always the case for the control, which has none).
    if flag.checklist_id not in case.checklist_ids:
        return False
    # 2. The flaw deleted a whole document: there is nothing left to point at.
    if case.removes_a_file:
        return True
    # 3. The flag's spot must match one of the case's spots.
    return overlaps(flag, case)


# ---------------------------------------------------------------------------------------------
# Scoring: one case, then all cases, then summary metrics
# ---------------------------------------------------------------------------------------------

def score_case(case: dict, flags: list[dict]) -> dict:
    """Score one case: ask hit() about each of the agent's flags and count.

    Args:
        case:  one entry of key["cases"] (JSON dict).
        flags: the agent's flags for this case (the "flags" list of runs/<run>/<case_id>.json).

    Each flag is a catch (the first hit), a duplicate (a later hit) or a false alarm, so
    n_flags == (1 if caught else 0) + duplicates + false_alarms.

    >>> case = {"case_id": "TP-021", "type": "inconsistent", "difficulty": "easy",
    ...         "checklist_ids": ["ops-designs"],
    ...         "locations": [{"file": "body.md", "section": "3.5", "page": 51, "role": "content"}]}
    >>> good = {"checklist_id": "ops-designs", "type": "inconsistent", "file": "body.md", "section": "3.5"}
    >>> bogus = {"checklist_id": "rec-revegetation", "type": "vague", "file": "body.md", "section": "6.4.4"}
    >>> row = score_case(case, [good, good, bogus])
    >>> {k: row[k] for k in ("caught", "n_flags", "duplicates", "false_alarms")}
    {'caught': True, 'n_flags': 3, 'duplicates': 1, 'false_alarms': 1}
    >>> score_case(case, [])["caught"]      # no flags: missed
    False
    """
    # Convert the JSON once; everything below works with typed objects.
    c = Case.from_dict(case)

    caught = False      # has any flag found the planted flaw yet?
    duplicates = 0      # flags that found the planted flaw after it was already caught
    false_alarms = []   # flags that matched nothing (kept as the original dicts, for scores.json)

    for f in (Flag.from_dict(d) for d in flags):
        if hit(f, c):
            if caught:
                duplicates += 1  # found again: neither helps nor hurts the score
            caught = True
        else:
            false_alarms.append(f.raw)

    # One row per case; aggregate() combines the rows into recall, precision, etc.
    return {
        "case_id": c.id,
        "type": c.type,
        "difficulty": c.difficulty,
        "planted": c.type != "none",              # False for the control; recall skips it
        "caught": caught,                         # did any flag find the planted flaw?
        "n_flags": len(flags),                    # all flags the agent raised
        "duplicates": duplicates,                 # extra catches after the first
        "false_alarms": len(false_alarms),        # flags that matched nothing
        "false_alarm_flags": false_alarms,        # those flags themselves, for inspection
    }


def aggregate(rows: list[dict]) -> dict:
    """Combine the per-case rows from score_case into summary metrics.

    recall                    caught planted cases / planted cases (the control is left out)
    precision                 caught cases / (all flags - duplicates)
    false_alarms_total        false alarms summed over every case
    false_alarms_per_case     that total / number of cases (control included)
    false_alarms_on_controls  false alarms on the clean control case(s) only
    by_type, by_difficulty    number of planted cases and recall for each type / difficulty

    Duplicates are taken out of the precision denominator, so they neither help nor hurt.
    Metrics with nothing to divide by are None.

    >>> def row(ftype, planted, caught, n_flags, duplicates, false_alarms):
    ...     return {"type": ftype, "difficulty": "easy" if planted else "none", "planted": planted,
    ...             "caught": caught, "n_flags": n_flags, "duplicates": duplicates,
    ...             "false_alarms": false_alarms}
    >>> s = aggregate([row("none", False, False, 1, 0, 1),
    ...                row("missing", True, True, 3, 1, 1),
    ...                row("vague", True, False, 0, 0, 0)])
    >>> s["recall"], s["precision"]        # 1 of 2 caught; 1 correct / (4 flags - 1 duplicate)
    (0.5, 0.333)
    >>> s["false_alarms_total"], s["false_alarms_on_controls"], s["false_alarms_per_case"]
    (2, 1, 0.67)
    """
    planted = [r for r in rows if r["planted"]]       # cases with a planted flaw
    controls = [r for r in rows if not r["planted"]]  # the clean control case(s)

    def recall(rs):
        # Share of these rows that were caught, or None if there are none.
        return round(sum(r["caught"] for r in rs) / len(rs), 3) if rs else None

    # Recall broken down by a field of the row, e.g. by("type") -> {"missing": {"n": 10, "recall": 0.8}, ...}
    by = lambda key: {k: {"n": len(v), "recall": recall(v)}
                      for k, v in sorted(_group(planted, key).items())}

    n_flags = sum(r["n_flags"] - r["duplicates"] for r in rows)  # every flag except duplicates
    correct = sum(r["caught"] for r in rows)                     # one per caught case
    return {
        "cases": len(rows),
        "planted_cases": len(planted),
        "recall": recall(planted),
        "precision": round(correct / n_flags, 3) if n_flags else None,
        "false_alarms_total": sum(r["false_alarms"] for r in rows),
        "false_alarms_per_case": round(sum(r["false_alarms"] for r in rows) / len(rows), 2) if rows else None,
        "false_alarms_on_controls": sum(r["false_alarms"] for r in controls),
        "by_type": by("type"),
        "by_difficulty": by("difficulty"),
    }


def _group(rows, key):
    """Group rows by the value of one field.

    >>> dict(_group([{"type": "missing", "id": 1}, {"type": "vague", "id": 2},
    ...              {"type": "missing", "id": 3}], "type"))
    {'missing': [{'type': 'missing', 'id': 1}, {'type': 'missing', 'id': 3}], 'vague': [{'type': 'vague', 'id': 2}]}
    """
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


# ---------------------------------------------------------------------------------------------
# Where the flags come from: an agent's run folder, or the answer key itself (oracle)
# ---------------------------------------------------------------------------------------------

def load_reports(runs_dir: Path, case_ids: list[str]) -> dict[str, list[dict]]:
    """Read an agent's flags for each case from <runs_dir>/<case_id>.json.

    Each report file looks like {"flags": [flag, flag, ...], ...} (see
    bench/schema/report.schema.json); only "flags" is used. A case with no report file, or a
    report with no "flags", gets [] and so counts as missed.
    """
    out = {}
    for cid in case_ids:
        p = runs_dir / f"{cid}.json"
        # Missing file -> empty list of flags.
        out[cid] = json.loads(p.read_text()).get("flags", []) if p.exists() else []
    return out


def oracle_reports(key: dict) -> dict[str, list[dict]]:
    """Perfect reports built from the answer key: one correct flag per planted flaw, none on controls.

    Scoring these should give recall 1.0, precision 1.0 and no false alarms; if not, the scorer
    or the answer key is broken. Each flag uses the case's first checklist item and its first
    edited location (skipping "related" and "toc" ones, which were not edited).

    >>> key = {"cases": [
    ...     {"case_id": "TP-000", "type": "none", "checklist_ids": [], "locations": [],
    ...      "expected_finding": ""},
    ...     {"case_id": "TP-021", "type": "inconsistent", "checklist_ids": ["ops-designs"],
    ...      "expected_finding": "West WRSF capacity conflicts within §3.5.",
    ...      "locations": [
    ...          {"file": "body.md", "section": "3.5", "page": 51, "role": "content"},
    ...          {"file": "body.md", "section": "3.5", "page": 51, "role": "related"}]}]}
    >>> reports = oracle_reports(key)
    >>> reports["TP-000"]
    []
    >>> reports["TP-021"]
    [{'checklist_id': 'ops-designs', 'type': 'inconsistent', 'file': 'body.md', 'section': '3.5', 'pages': [51], 'quote': None, 'explanation': 'West WRSF capacity conflicts within §3.5.'}]
    """
    out = {}
    for c in key["cases"]:
        # The clean control: a perfect agent raises nothing.
        if c["type"] == "none":
            out[c["case_id"]] = []
            continue
        # Prefer a location that was actually edited; fall back to any location, then to none.
        content = [l for l in c["locations"] if l.get("role") not in ("toc", "related")]
        loc = (content or c["locations"] or [{}])[0]
        # One flag in the report format; "pages" is a list, built from the location's single page.
        out[c["case_id"]] = [{
            "checklist_id": c["checklist_ids"][0], "type": c["type"],
            "file": loc.get("file"), "section": loc.get("section"),
            "pages": [loc["page"]] if loc.get("page") else [], "quote": None,
            "explanation": c["expected_finding"],
        }]
    return out


def score(key: dict, reports: dict[str, list[dict]]) -> dict:
    """Score every case in the answer key.

    Args:
        key:     the answer key (bench/answer_key/<plan>.json).
        reports: {case_id: [flag, ...]} from load_reports or oracle_reports. A case missing
                 from it gets no flags.

    Returns {"summary": aggregate()'s metrics, "cases": score_case()'s row for each case}.

    >>> key = {"plan": "demo", "known_gaps": [], "cases": [
    ...     {"case_id": "TP-000", "type": "none", "difficulty": "none", "checklist_ids": [],
    ...      "locations": [], "expected_finding": ""},
    ...     {"case_id": "TP-021", "type": "inconsistent", "difficulty": "easy",
    ...      "checklist_ids": ["ops-designs"], "expected_finding": "",
    ...      "locations": [{"file": "body.md", "section": "3.5", "page": 51, "role": "content"}]}]}
    >>> res = score(key, oracle_reports(key))
    >>> res["summary"]["recall"], res["summary"]["precision"], res["summary"]["false_alarms_total"]
    (1.0, 1.0, 0)
    >>> score(key, {})["summary"]["recall"]      # no reports at all
    0.0
    """
    # One row per case; a case with no report is scored with no flags.
    rows = [score_case(c, reports.get(c["case_id"], [])) for c in key["cases"]]
    return {"summary": aggregate(rows), "cases": rows}


# ---------------------------------------------------------------------------------------------
# Output and command line
# ---------------------------------------------------------------------------------------------

def to_markdown(res: dict, title: str) -> str:
    """Format score()'s result as a Markdown report (written to scores.md and printed).

    Sections: the headline metrics, recall by type, recall by difficulty, and one row per case.
    In the per-case table "Caught" is yes / no for planted cases and "—" for the control.
    """
    s = res["summary"]
    lines = [f"# Scores: {title}", "",
             "| Metric | Value |", "|---|---|",
             f"| Recall | {s['recall']} |",
             f"| Precision | {s['precision']} |",
             f"| False alarms (total) | {s['false_alarms_total']} |",
             f"| False alarms on clean control | {s['false_alarms_on_controls']} |",
             "", "## Recall by type", "", "| Type | Cases | Recall |", "|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {v['recall']} |" for k, v in s["by_type"].items()]
    lines += ["", "## Recall by difficulty", "", "| Difficulty | Cases | Recall |", "|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {v['recall']} |" for k, v in s["by_difficulty"].items()]
    # One row per case: "yes"/"no" for planted cases, "—" for the control.
    lines += ["", "## Per case", "", "| Case | Type | Difficulty | Caught | Flags | False alarms |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['case_id']} | {r['type']} | {r['difficulty']} | {'yes' if r['caught'] else ('—' if not r['planted'] else 'no')} | {r['n_flags']} | {r['false_alarms']} |"
              for r in res["cases"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Command line entry point: score a run folder (or the oracle) for one plan.

        uv run python -m bench.score thacker_pass_clean runs/<run_name>   # writes scores.json + scores.md there
        uv run python -m bench.score thacker_pass_clean --oracle          # prints only

    The plan name picks bench/answer_key/<plan>.json. It must be a plan with no known gaps.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("runs_dir", nargs="?", type=Path)
    ap.add_argument("--oracle", action="store_true", help="score perfect reports built from the answer key")
    args = ap.parse_args()

    # Load the answer key for the chosen plan.
    key = json.loads((ROOT / "bench" / "answer_key" / f"{args.plan}.json").read_text())

    # Known gaps are real flaws in the published plan. This scorer has no notion of them, so a
    # flag on one would count as a false alarm. Only plans with none (thacker_pass_clean) are allowed.
    n_gaps = len(key.get("known_gaps") or [])
    assert n_gaps == 0, (
        f"{key.get('plan')} has {n_gaps} known gaps, but this scorer only supports plans with "
        f"none. Use thacker_pass_clean.")

    # Get the flags to score: the oracle's perfect ones, or the agent's from the run folder.
    if args.oracle:
        reports, title = oracle_reports(key), "oracle"
    elif args.runs_dir:
        reports, title = load_reports(args.runs_dir, [c["case_id"] for c in key["cases"]]), str(args.runs_dir)
    else:
        ap.error("give a runs_dir or --oracle")

    res = score(key, reports)
    md = to_markdown(res, title)

    # Save the results next to the reports when a run folder was given (note: with --oracle
    # AND a runs_dir, the oracle's scores are what gets written there).
    if args.runs_dir:
        (args.runs_dir / "scores.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        (args.runs_dir / "scores.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
