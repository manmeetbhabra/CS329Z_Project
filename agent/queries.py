"""Search words for each checklist requirement (knowledge/item_queries.yaml).

    uv run python -m agent.queries --write     # once: ask the model for search words, write the file
    load_item_queries()["rec-riparian"]        # -> ["Riparian mitigation. operations may affect …",
                                               #     "riparian area mitigation", "springs and seeps", …]

Why: the checklist uses the regulation's words ("Riparian mitigation."), while a plan uses its own
("five seasonal springs", "seeps"). BM25 only matches shared words, so better words mean better
passages for the judge. V0's search check measures the gain:
    uv run python -m harness.probe --queries item_queries

Fair: WriteQueries sees ONLY the checklist item, never a plan, so the words are the same for every
plan (the held-out ones included) and can't leak a planted flaw. The item's own text is always
query #1, and a person reviews the file before it is used.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import dspy
import yaml

from agent.checklist import load_checklist

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ROOT / "knowledge" / "item_queries.yaml"


class WriteQueries(dspy.Signature):
    """You help search a mining Plan of Operations submitted to the U.S. Bureau of Land Management
    (43 CFR 3809). For ONE requirement, write the short search phrases such a plan would use
    where it addresses that requirement: the section titles, technical terms and synonyms a mining
    plan actually uses, not the regulation's wording. Each phrase is 2 to 5 words. Wrap a phrase in
    double quotes only if those exact words must appear (e.g. a standard term like "factor of safety")."""
    requirement: str = dspy.InputField(desc="one checklist item: id, citation, requirement, applies when, too vague if")
    queries: list[str] = dspy.OutputField(desc="4 to 8 short search phrases")


def make_queries(path: Path = QUERIES) -> dict[str, list[str]]:
    """Ask the configured LM for search words for every item and write them to path.
    -> {"op-identity": ["Name, mailing address and phone number of the operator(s).", "operator information", …], …}"""
    write = dspy.Predict(WriteQueries)
    items = load_checklist()
    results = write.batch([dspy.Example(requirement=i.render()).with_inputs("requirement") for i in items],
                          num_threads=8)
    out = {}
    for item, r in zip(items, results):
        words = [q.strip() for q in (getattr(r, "queries", None) or []) if q and q.strip()]
        out[item.id] = [item.query()] + [q for q in dict.fromkeys(words) if q.lower() != item.query().lower()][:8]
    header = ("# Search words per checklist item, written by agent/queries.py (WriteQueries, from the checklist\n"
              "# text only, never from a plan) and reviewed by a person. Query 1 is always the item's own text.\n"
              "# A phrase in double quotes is matched exactly (grep); anything else goes to BM25.\n")
    Path(path).write_text(header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    load_item_queries.cache_clear()
    return out


@lru_cache(maxsize=4)
def _read(path: Path) -> dict[str, list[str]]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def load_item_queries(path: Path = QUERIES) -> dict[str, list[str]]:
    """{item id: [queries]} for all 51 items. An item missing from the file gets [item.query()].
    load_item_queries()["rec-riparian"][0] == items_by_id()["rec-riparian"].query()"""
    saved = _read(Path(path))
    return {i.id: list(saved.get(i.id) or [i.query()]) for i in load_checklist()}


load_item_queries.cache_clear = _read.cache_clear  # make_queries refreshes what later loads see


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="call the model and (over)write knowledge/item_queries.yaml")
    args = ap.parse_args()
    if not args.write:
        for k, v in load_item_queries().items():
            print(k, v)
        return
    from agent.config import configure
    configure("dev")
    out = make_queries()
    print(f"wrote {QUERIES} ({len(out)} items, {sum(len(v) for v in out.values())} queries)")


if __name__ == "__main__":
    main()
