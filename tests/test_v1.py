"""Tests for V1 (search words, judge inputs, the reviewer, the harness path, miss labels).
Offline: the judge is DSPy's DummyLM with scripted answers keyed by checklist item.

Run:  uv run pytest -q
"""

import json
from pathlib import Path

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from agent.checklist import load_checklist
from agent.plan import Location, PlanIndex
from agent.queries import load_item_queries
from agent.search import PlanSearch
from agent.tools import render_for_judge

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "bench" / "cases" / "thacker_pass_clean"


# ----------------------------------------------------------------- search words
def test_item_queries_cover_every_item_and_start_with_its_own_text():
    queries = load_item_queries()
    items = load_checklist()
    assert set(queries) == {i.id for i in items}
    assert all(queries[i.id][0] == i.query() for i in items)
    assert all(2 <= len(queries[i.id]) <= 9 for i in items)


def test_missing_items_fall_back_to_their_own_text(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text("op-tin: [only this one]\n")
    q = load_item_queries(path)
    assert q["op-tin"] == ["only this one"]
    assert q["rec-riparian"] == [next(i for i in load_checklist() if i.id == "rec-riparian").query()]


# ----------------------------------------------------------------- judge inputs
@pytest.fixture(scope="module")
def search000():
    return PlanSearch(PlanIndex.load(CASES / "TP-000"))


def test_render_for_judge_numbers_heads_and_caps_passages(search000):
    passages = search000.search("spill contingency plan", k=3)
    text = render_for_judge(passages, search000, max_chars=300)
    assert text.startswith("[1] appendix_E.md")
    assert "[3] " in text and "[4] " not in text
    bodies = [block.split("\n", 1)[1].split("\nSub-sections: ")[0] for block in text.split("\n\n[")[1:]]
    assert all(len(b) <= 330 for b in bodies)        # the cap applies to the passage text itself


def test_passages_list_their_subsections(search000):
    from agent.tools import subsections
    subs = subsections(search000.index, "body.md", "6.5")
    assert subs[0].startswith("6.5.1 Decommissioning of Facilities")
    intro = next(p for p in search000.search("facility reclamation practices decommissioning", k=10) if p.section == "6.5")
    assert "Sub-sections: 6.5.1 Decommissioning of Facilities" in render_for_judge([intro], search000)


def test_plan_map_lists_real_headings_only():
    from agent.tools import plan_map
    tp001 = plan_map(PlanIndex.load(CASES / "TP-001"))            # Appendix E deleted
    assert "3.10 Process Plant Site (p.62)" in tp001
    assert "appendix_E.md" not in tp001 and "appendix_D.md" in tp001


def test_heading_only_passages_bring_their_subsections(search000):
    wildlife = next(p for p in search000.search("wildlife baseline surveys", k=20) if p.heading == "2.3.6 Wildlife")
    assert "General Wildlife" in search000.passage_text(wildlife)


# ----------------------------------------------------------------- the reviewer, scripted
def _verdict(status="present", quote="", file="", section="", confidence=0.9, reason="scripted"):
    """A scripted JudgeItem answer: 'present'/'not_applicable', or one finding of type `status`."""
    if status in ("present", "not_applicable"):
        return {"reasoning": "scripted", "verdict": status, "findings": [], "reason": reason, "confidence": confidence}
    finding = {"status": status, "quote": quote, "file": file, "section": section, "reason": reason,
               "confidence": confidence}
    return {"reasoning": "scripted", "verdict": "problem", "findings": [finding], "reason": reason,
            "confidence": confidence}


def scripted_judge(special: dict) -> DummyLM:
    """A DummyLM answering for every checklist item: 'present' unless `special` says otherwise.
    Keys are '<item id> (' which appears once, at the start of that item's requirement text."""
    answers = {f"{i.id} (": special.get(i.id, _verdict()) for i in load_checklist()}
    return DummyLM(answers, adapter=dspy.JSONAdapter())


SPECIAL = {
    "ops-designs": _verdict("inconsistent", quote="| West WRSF | 37.2 | 5,252 |", file="body.md", section="3.4"),
    "rec-drill-holes": _verdict("missing", quote="", file="body.md", section="6.8"),
    "rec-topsoil": _verdict("vague", quote="growth media", confidence=0.2),          # too unsure: dropped
    "pf-exclusion": _verdict("not_applicable"),
}


def test_reviewer_turns_verdicts_into_placed_flags():
    from agent.systems.v1 import PlanReviewer
    with dspy.context(lm=scripted_judge(SPECIAL), adapter=dspy.JSONAdapter()):
        pred = PlanReviewer(threads=4)(case_dir=str(CASES / "TP-021"))
    by_item = {f["checklist_id"]: f for f in pred.flags}
    assert set(by_item) == {"ops-designs", "rec-drill-holes"}
    assert by_item["ops-designs"]["anchor"] == "quote"
    assert (by_item["ops-designs"]["section"], by_item["ops-designs"]["pages"]) == ("3.5", [51])
    assert by_item["rec-drill-holes"]["anchor"] == "section" and by_item["rec-drill-holes"]["section"] == "6.8"
    assert len(pred.verdicts) == 51 and pred.report["case_id"] == "TP-021"
    assert {v["item"]: v["status"] for v in pred.verdicts}["pf-exclusion"] == "not_applicable"


def test_v1_through_the_harness_offline(tmp_path):
    from harness.run import run
    summary = run("v1", cases="TP-021", name="v", runs_root=tmp_path, threads=1, quiet=True,
                  lm=scripted_judge(SPECIAL))
    assert summary["caught"] == 1 and summary["errors"] == 0 and summary["lm_calls"] == 51
    verdicts = (tmp_path / "v" / "verdicts" / "TP-021.jsonl").read_text().splitlines()
    assert len(verdicts) == 51 and json.loads(verdicts[0])["passages"]
    assert summary["params"]["min_confidence"] == 0.5


# ----------------------------------------------------------------- miss labels
def _row(item, status, confidence=0.9, places=()):
    return {"item": item, "status": status, "confidence": confidence, "reason": "",
            "passages": [{"file": f, "section": s, "pages": list(p)} for f, s, p in places]}


def test_each_miss_gets_the_right_label():
    from harness.data import load_devset
    from harness.misses import label_miss
    gold = load_devset("thacker_pass_clean", "TP-021")[0].gold          # ops-designs, body.md §3.5 p.51
    at_flaw = [("body.md", "6.4", (108,)), ("body.md", "3.5", (51,))]
    elsewhere = [("appendix_C.md", "3.2", (13,))]
    assert label_miss(gold, [_row("ops-designs", "vague", places=elsewhere)], 0.5)["label"] == "search miss"
    assert label_miss(gold, [_row("ops-designs", "present", places=at_flaw)], 0.5)["label"] == "judge miss"
    assert label_miss(gold, [_row("ops-designs", "vague", 0.3, at_flaw)], 0.5)["label"] == "threshold miss"
    hit = label_miss(gold, [_row("ops-designs", "inconsistent", 0.9, at_flaw)], 0.5)
    assert hit["label"] == "placing miss" and hit["seen_rank"] == 2
