"""Why did a run miss what it missed? One label per missed planted flaw, plus the false alarms.

    uv run python -m harness.misses runs/v1-smoke        # -> runs/v1-smoke/misses.md

For each planted flaw the run did not catch, look at the judge's verdict for the flaw's
checklist item(s) (runs/<name>/verdicts/TP-###.jsonl) and ask:

  Q1  Was the flaw's place among the passages the judge saw for that item?
      (For a whole deleted file: did missing_refs() point at it?)       no  -> search miss
  Q2  What did the judge say?   present / not_applicable / error        -> judge miss
                                vague / missing, below min_confidence   -> threshold miss
                                vague / missing, flagged                -> placing miss
                                                                           (the flag's place or item
                                                                            didn't match the scorer's)
A label tells us what to work on: search words / k, the judge's prompt, the threshold, the locator.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import dspy

from agent.plan import Location, PlanIndex
from agent.search import PlanSearch
from agent.systems.v1 import PROBLEMS
from harness.data import load_devset
from bench import score as bench_score
from harness.metric import gold_case, score_row, search_hit

ROOT = Path(__file__).resolve().parents[1]


def label_miss(gold: dict, verdicts: list[dict], min_confidence: float, case_dir: str | None = None) -> dict:
    """The label for one missed flaw.
    -> {"label": "judge miss", "item": "mon-details", "status": "present", "confidence": 0.9, "seen_rank": 3,
        "reason": "..."}"""
    case = gold_case(gold)
    rows = [v for v in verdicts if v["item"] in case.checklist_ids]
    if not rows:
        return {"label": "search miss", "item": case.checklist_ids[0], "status": "-", "confidence": 0.0,
                "seen_rank": None, "reason": "no verdict for the flaw's item"}

    def seen(v):
        if case.removes_a_file:
            return 1 if _deleted_file_is_referenced(case, case_dir) else None
        return search_hit([Location(p["file"], p["section"], tuple(p["pages"])) for p in v["passages"]], gold)

    # prefer the verdict that got furthest: seen and flagged > seen > anything
    scored = sorted(rows, key=lambda v: (seen(v) is None, v["status"] not in PROBLEMS))
    v = scored[0]
    rank = seen(v)
    if rank is None:
        label = "search miss"
    elif v["status"] not in PROBLEMS:
        label = "judge miss"
    elif max(_finding_confidences(v)) < min_confidence:
        label = "threshold miss"
    else:
        label = "placing miss"
    return {"label": label, "item": v["item"], "status": v["status"], "confidence": v["confidence"],
            "seen_rank": rank, "reason": v.get("reason", ""), "placed": v.get("placed")}


def _finding_confidences(v: dict) -> list[float]:
    """Confidence of each finding (V1.1 rows), or the verdict's own confidence (V1.0 rows)."""
    return [f.get("confidence", 0.0) for f in v.get("findings") or []] or [v.get("confidence", 0.0)]


def _deleted_file_is_referenced(case, case_dir: str | None) -> bool:
    if not case_dir:
        return False
    deleted = {l.file for l in case.locations}
    return any(r.target in deleted for r in PlanSearch(PlanIndex.load(case_dir)).missing_refs())


def analyse(run_dir: Path, plan: str = "thacker_pass_clean") -> dict:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "run.json").read_text())
    min_conf = (summary.get("params") or {}).get("min_confidence", 0.5)
    misses, alarms, control, catches = [], defaultdict(list), [], []
    fa_keys = []                       # (case_id, flag key) of every false alarm
    control_path = run_dir / "TP-000.json"
    control_keys = ({_flag_key(f) for f in json.loads(control_path.read_text())["flags"]}
                    if control_path.exists() else None)
    for ex in load_devset(plan, summary.get("cases", "all")):
        report_path = run_dir / f"{ex.case_id}.json"
        if not report_path.exists():
            continue
        flags = json.loads(report_path.read_text())["flags"]
        vpath = run_dir / "verdicts" / f"{ex.case_id}.jsonl"
        verdicts = [json.loads(l) for l in vpath.read_text().splitlines()] if vpath.exists() else []
        row = score_row(ex, dspy.Prediction(flags=flags))
        for f in row["false_alarm_flags"]:
            alarms[f["checklist_id"]].append(ex.case_id)
            fa_keys.append((ex.case_id, _flag_key(f)))
            if ex.case_id == "TP-000":
                control.append(f)
        if row["planted"] and not row["caught"]:
            m = label_miss(ex.gold, verdicts, min_conf, ex.case_dir)
            misses.append({"case_id": ex.case_id, "type": row["type"], "difficulty": row["difficulty"], **m})
        if row["planted"] and row["caught"]:
            case = gold_case(ex.gold)
            hits = [f for f in flags if bench_score.hit(bench_score.Flag.from_dict(f), case)]
            lucky = control_keys is not None and all(_flag_key(f) in control_keys for f in hits)
            catches.append({"case_id": ex.case_id, "type": row["type"], "difficulty": row["difficulty"],
                            "lucky": lucky, "flag": {k: hits[0].get(k) for k in ("checklist_id", "type", "file", "section", "quote", "path")}})
    genuine = [c for c in catches if not c["lucky"]]
    return {"false_alarms": false_alarm_counts(fa_keys, control_keys),
            "run": str(run_dir), "min_confidence": min_conf, "misses": misses,
            "labels": dict(Counter(m["label"] for m in misses)), "false_alarms_by_item": dict(alarms),
            "control_flags": control, "catches": catches,
            "genuine_caught": len(genuine), "lucky_caught": len(catches) - len(genuine)}


def false_alarm_counts(fa_keys: list[tuple], control_keys: set | None) -> dict:
    """Split false alarms into standing ones (the same flag as on the clean control, repeated in every case)
    and case-specific ones, and count distinct places - what a user reviewing one plan would see.
    [("TP-000", k1), ("TP-001", k1), ("TP-001", k2)], control {k1}
      -> {"total": 3, "control": 1, "standing": 1, "case_specific": 1, "distinct": 2}"""
    flawed = [(c, k) for c, k in fa_keys if c != "TP-000"]
    standing = sum(1 for _, k in flawed if control_keys and k in control_keys)
    return {"total": len(fa_keys), "control": sum(1 for c, _ in fa_keys if c == "TP-000"),
            "standing": standing, "case_specific": len(flawed) - standing,
            "distinct": len({k for _, k in fa_keys})}


def _flag_key(f: dict) -> tuple:
    """A flag's identity across cases: same item, file and section (pages can shift by an edit)."""
    return (f.get("checklist_id"), f.get("file"), f.get("section"))


def to_markdown(res: dict) -> str:
    out = [f"# Misses: {res['run']}", "", f"min_confidence = {res['min_confidence']}", "",
           "## Catches", "",
           f"{res['genuine_caught'] + res['lucky_caught']} caught: **{res['genuine_caught']} genuine**, "
           f"{res['lucky_caught']} lucky (the hitting flag is identical - same item, file, section - to a flag this "
           "run also raised on the clean control TP-000, so it isn't evidence the flaw was found).", "",
           "| case | type | difficulty | lucky? | item | place | quote |", "|---|---|---|---|---|---|---|"]
    for c in res["catches"]:
        f = c["flag"]
        out.append(f"| {c['case_id']} | {c['type']} | {c['difficulty']} | {'lucky' if c['lucky'] else 'genuine'} | "
                   f"{f['checklist_id']} | {f['file']} §{f['section'] or '-'} | {(f['quote'] or '').replace('|', '/')[:70]} |")
    out += ["", "## Missed planted flaws by label", "", "| label | count |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in sorted(res["labels"].items(), key=lambda kv: -kv[1])] or ["| (none) | 0 |"]
    out += ["", "| case | type | difficulty | item | label | judge said | conf | seen at | reason |",
            "|---|---|---|---|---|---|---|---|---|"]
    for m in res["misses"]:
        reason = (m["reason"] or "").replace("|", "/").replace("\n", " ")[:140]
        out.append(f"| {m['case_id']} | {m['type']} | {m['difficulty']} | {m['item']} | **{m['label']}** | "
                   f"{m['status']} | {m['confidence']:.2f} | {m['seen_rank'] or '-'} | {reason} |")
    fa = res["false_alarms"]
    out += ["", "## False alarms, deduplicated", "",
            "Most false alarms are the clean control's flags repeated in every case. Read precision with these:", "",
            "| | count |", "|---|---|",
            f"| total (what bench precision counts) | {fa['total']} |",
            f"| on the clean control TP-000 | {fa['control']} |",
            f"| in flawed cases, repeating a TP-000 flag (standing) | {fa['standing']} |",
            f"| in flawed cases, new (case-specific) | {fa['case_specific']} |",
            f"| distinct places (item, file, section) across all cases | {fa['distinct']} |"]
    out += ["", "## False alarms by checklist item", "", "| item | false alarms | cases |", "|---|---|---|"]
    for item, cases in sorted(res["false_alarms_by_item"].items(), key=lambda kv: -len(kv[1])):
        out.append(f"| {item} | {len(cases)} | {', '.join(sorted(set(cases)))} |")
    out += ["", f"## Flags on the clean control TP-000 ({len(res['control_flags'])})", "",
            "| item | type | place | quote |", "|---|---|---|---|"]
    for f in res["control_flags"]:
        place = f"{f.get('file')} §{f.get('section') or '-'} p.{','.join(map(str, f.get('pages') or []))}"
        quote = (f.get("quote") or "").replace("|", "/").replace("\n", " ")[:90]
        out.append(f"| {f['checklist_id']} | {f['type']} | {place} | {quote} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--plan", default="thacker_pass_clean")
    args = ap.parse_args()
    res = analyse(args.run_dir, args.plan)
    md = to_markdown(res)
    (args.run_dir / "misses.md").write_text(md, encoding="utf-8")
    (args.run_dir / "misses.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
