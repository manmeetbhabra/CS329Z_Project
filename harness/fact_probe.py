"""Did the value path see the planted value? Measures V2's ledger before blaming its judges.

    uv run python -m harness.fact_probe runs/v2v-smoke        # -> runs/v2v-smoke/facts.md

For each inconsistent or implausible flaw in a run (from runs/<name>/values/TP-###.json):
  extracted   a fact at the flaw's edited place whose value matches the planted (replacement) value
  local       CheckChunk reported a finding in the edited chunk
  grouped     (inconsistent) the planted fact shares a disagreeing group with a copy at a 'related' place
  conflict    that group was judged a real conflict
  caught      the official rule caught the flaw
and one label for each miss:
  extraction miss  -> the planted value never made it into the ledger
  grouping miss    -> extracted, but never compared with its other copy (inconsistent only)
  value-judge miss -> seen (locally, or in a group) but judged fine
  item/placing miss-> flagged, but with a checklist item or place the scorer didn't accept
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import dspy

from agent.facts import parse_value
from bench import score as bench_score
from harness.data import load_devset
from harness.metric import score_row

ROOT = Path(__file__).resolve().parents[1]
VALUE_TYPES = ("inconsistent", "implausible")


def _numbers(text: str) -> list[float]:
    """Every number in a text, words included: "six to 12 inches; 0.12 lb" -> [6, 12, 0.12]"""
    out = []
    for piece in re.split(r"[;|]", text or ""):
        v = parse_value(piece)
        if v:
            out += [v[0], v[1]]
    out += [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text or "")]
    return out


def _matches(value: str, numbers: list[float]) -> bool:
    v = parse_value(value)
    return bool(v) and any(abs(x - n) <= 0.01 * max(abs(x), abs(n), 1e-9) for x in v for n in numbers)


def _at(fact: dict, loc: dict) -> bool:
    """Is the fact at this answer-key location (the scorer's same_place rule)?"""
    a = bench_score.Location(fact.get("file"), fact.get("section"), tuple(fact.get("pages") or ()))
    b = bench_score.Location(loc.get("file"), loc.get("section"), (loc["page"],) if loc.get("page") else ())
    return bench_score.same_place(a, b)


def probe_case(gold: dict, log: dict, caught: bool) -> dict:
    content = [l for l in gold["locations"] if l.get("role") == "content"]
    related = [l for l in gold["locations"] if l.get("role") == "related"]
    planted = _numbers(" ".join(l.get("replacement") or "" for l in content))
    planted_facts = [f for f in log.get("facts", []) if any(_at(f, l) for l in content) and _matches(f["value"], planted)]
    local = [f for f in log.get("local", []) if any(_at({**f, "pages": (f.get("placed") or {}).get("pages", [])}, l) for l in content)]
    grouped = conflict = False
    for g in log.get("groups", []):
        has_planted = any(any(_at(s, l) for l in content) and _matches(s["value"], planted) for s in g["statements"])
        has_other = any(any(_at(s, l) for l in related) for s in g["statements"]) if related else len(g["statements"]) > 1
        if has_planted and has_other:
            grouped = True
            conflict = conflict or bool(g.get("conflict"))
    row = {"extracted": bool(planted_facts), "local": bool(local), "grouped": grouped, "conflict": conflict,
           "caught": caught, "planted_values": sorted(set(planted))[:6],
           "facts_at_flaw": [f"{f['subject']} | {f['attribute']} | {f['value']} {f['unit']}" for f in planted_facts][:3]}
    if caught:
        row["label"] = "caught"
    elif not planted_facts and not local:
        row["label"] = "extraction miss"
    elif gold["type"] == "inconsistent" and not local and not grouped:
        row["label"] = "grouping miss"
    elif local or conflict:
        row["label"] = "item/placing miss"
    else:
        row["label"] = "value-judge miss"
    return row


def probe_run(run_dir: Path, plan: str = "thacker_pass_clean") -> list[dict]:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "run.json").read_text())
    rows = []
    for ex in load_devset(plan, summary.get("cases", "all")):
        if ex.gold["type"] not in VALUE_TYPES:
            continue
        vpath, rpath = run_dir / "values" / f"{ex.case_id}.json", run_dir / f"{ex.case_id}.json"
        if not vpath.exists() or not rpath.exists():
            continue
        caught = score_row(ex, dspy.Prediction(flags=json.loads(rpath.read_text())["flags"]))["caught"]
        rows.append({"case_id": ex.case_id, "type": ex.gold["type"], "difficulty": ex.gold["difficulty"],
                     **probe_case(ex.gold, json.loads(vpath.read_text()), caught)})
    return rows


def to_markdown(rows: list[dict], run: str) -> str:
    n = len(rows) or 1
    out = [f"# Fact probe: {run}", "", f"{len(rows)} value flaws (inconsistent + implausible).", "",
           "| step | share |", "|---|---|",
           f"| planted value extracted as a fact | {sum(r['extracted'] for r in rows)}/{len(rows)} |",
           f"| CheckChunk flagged the edited chunk | {sum(r['local'] for r in rows)}/{len(rows)} |",
           f"| grouped with its other copy (inconsistent) | {sum(r['grouped'] for r in rows if r['type'] == 'inconsistent')}"
           f"/{sum(r['type'] == 'inconsistent' for r in rows)} |",
           f"| caught | {sum(r['caught'] for r in rows)}/{len(rows)} |", "",
           "| label | count |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in Counter(r["label"] for r in rows).most_common()]
    out += ["", "| case | type | diff | extracted | local | grouped | conflict | label | facts at the flaw |",
            "|---|---|---|---|---|---|---|---|---|"]
    yn = lambda b: "yes" if b else "-"
    for r in rows:
        out.append(f"| {r['case_id']} | {r['type']} | {r['difficulty']} | {yn(r['extracted'])} | {yn(r['local'])} | "
                   f"{yn(r['grouped'])} | {yn(r['conflict'])} | **{r['label']}** | {'; '.join(r['facts_at_flaw'])[:110]} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--plan", default="thacker_pass_clean")
    args = ap.parse_args()
    rows = probe_run(args.run_dir, args.plan)
    md = to_markdown(rows, str(args.run_dir))
    (args.run_dir / "facts.md").write_text(md, encoding="utf-8")
    (args.run_dir / "facts.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
