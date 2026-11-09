"""The search check: can search find the planted problems? No AI involved.

    uv run python -m harness.probe --name probe-v0                  # queries = checklist item text
    uv run python -m harness.probe --name probe-v0 --k 1 3 5 10 20

For every flawed case (except those that delete a whole file: there is nothing left to find),
search for the planted flaw's checklist item and ask: is a top-k passage at the flaw's place?
"At the place" uses the scorer's own rule (bench.score.same_place). A case with two checklist
items counts as found if either item's search finds it.

Missing flaws are a special case: the content was deleted, so the "place" is where it used to be.
Search can only land there through what's left around it, so their recall is reported separately.

Writes runs/<name>/recall.md (tables) and rows.jsonl (one row per case and item).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from agent.checklist import items_by_id
from agent.plan import PlanIndex
from agent.search import PlanSearch
from harness.data import DEFAULT_PLAN, load_devset
from harness.metric import gold_case, search_hit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KS = (1, 3, 5, 10, 20)


def queries_for(item, mode: str) -> list[str]:
    """item_text: [the item's own text] (V0).  item_queries: the saved search words (V1)."""
    if mode == "item_text":
        return [item.query()]
    if mode == "item_queries":
        from agent.queries import load_item_queries
        return load_item_queries()[item.id]
    raise ValueError(f"unknown query mode {mode!r}: use item_text or item_queries")


def probe(plan: str = DEFAULT_PLAN, mode: str = "item_text", ks=DEFAULT_KS) -> list[dict]:
    items = items_by_id()
    rows = []
    for ex in load_devset(plan, "flawed"):
        case = gold_case(ex.gold)
        if case.removes_a_file:
            continue
        search = PlanSearch(PlanIndex.load(ex.case_dir))
        for item_id in case.checklist_ids:
            passages = search.hybrid(queries_for(items[item_id], mode), k=max(ks))
            rank = search_hit([p.location for p in passages], ex.gold)
            rows.append({"case_id": case.id, "type": case.type, "difficulty": case.difficulty,
                         "item": item_id, "rank": rank,
                         "top3": [p.header() for p in passages[:3]]})
    return rows


def per_case(rows: list[dict]) -> list[dict]:
    """One row per case: the best rank over its checklist items."""
    best: dict[str, dict] = {}
    for r in rows:
        b = best.get(r["case_id"])
        if b is None or (r["rank"] is not None and (b["rank"] is None or r["rank"] < b["rank"])):
            best[r["case_id"]] = r
    return list(best.values())


def recall_table(cases: list[dict], ks, group: str | None) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        groups[c[group] if group else "all"].append(c)
    order = {"missing": 0, "vague": 1, "inconsistent": 2, "implausible": 3, "easy": 0, "medium": 1, "subtle": 2}
    head = f"| {group or 'cases'} | n | " + " | ".join(f"recall@{k}" for k in ks) + " |"
    lines = [head, "|" + "---|" * (len(ks) + 2)]
    for g in sorted(groups, key=lambda g: (order.get(g, 9), g)):
        cs = groups[g]
        cells = [f"{sum(c['rank'] is not None and c['rank'] <= k for c in cs) / len(cs):.2f}" for k in ks]
        lines.append(f"| {g} | {len(cs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_markdown(rows: list[dict], ks, mode: str, plan: str) -> str:
    cases = per_case(rows)
    misses = [c for c in cases if c["rank"] is None or c["rank"] > 5]
    out = [f"# Search check: {plan}, queries = {mode}", "",
           f"{len(cases)} flawed cases searched (whole-file deletions left out). A case is found at k "
           "if a top-k passage for its checklist item is at the planted flaw's place (bench.score.same_place).", "",
           "## Overall", "", recall_table(cases, ks, None), "",
           "## By flaw type", "", recall_table(cases, ks, "type"), "",
           "## By difficulty", "", recall_table(cases, ks, "difficulty"), "",
           "## Not found in the top 5", "", "| case | type | item | rank | top passage |", "|---|---|---|---|---|"]
    for c in sorted(misses, key=lambda c: c["case_id"]):
        out.append(f"| {c['case_id']} | {c['type']} | {c['item']} | {c['rank'] or '-'} | {c['top3'][0] if c['top3'] else '-'} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--queries", default="item_text", help="item_text (V0); item_queries comes with V1")
    ap.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    ap.add_argument("--name", default="probe-v0")
    args = ap.parse_args()
    rows = probe(args.plan, args.queries, tuple(args.k))
    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    md = to_markdown(rows, tuple(args.k), args.queries, args.plan)
    (out / "recall.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
