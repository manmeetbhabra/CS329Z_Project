"""Tests for V2: the fact ledger (agent/facts.py), the value path and merge (agent/systems/v2.py),
and the fact probe. Offline: the model calls are scripted.

Run:  uv run pytest -q
"""

from pathlib import Path

import dspy
import pytest

from agent.facts import (Fact, PlacedFact, disagreements, group_facts, normalize_unit, numeric_chunks, parse_value,
                         table_total_checks)
from agent.plan import PAGE_MARKER, PlanIndex
from bench import score as bench_score
from harness.data import load_devset

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "bench" / "cases" / "thacker_pass_clean"


# ----------------------------------------------------------------- values and units
@pytest.mark.parametrize("text,expected", [
    ("5,545.0", (5545.0, 5545.0)), ("six to 12", (6.0, 12.0)), ("18–24", (18.0, 24.0)),
    ("2.5H:1V", (2.5, 2.5)), ("3:1", (3.0, 3.0)), ("80 million", (8e7, 8e7)), ("one-half to one inch", (0.5, 1.0)),
    ("0.12", (0.12, 0.12)), ("Year 44 and Year 45", (44.0, 45.0)), ("none", None)])
def test_parse_value(text, expected):
    assert parse_value(text) == expected


@pytest.mark.parametrize("unit,expected", [
    ("M CY", ("mcy", 1.0)), ("million cubic yards", ("mcy", 1.0)), ("yrs", ("yr", 1.0)),
    ("million gallons", ("gal", 1e6)), ("lb PLS/acre", ("lb_pls_acre", 1.0)), ("degrees", ("deg", 1.0)), ("", ("", 1.0))])
def test_normalize_unit(unit, expected):
    assert normalize_unit(unit) == expected


# ----------------------------------------------------------------- grouping
def _pf(subject, attribute, value, unit, section, quote=None):
    return PlacedFact(Fact(subject=subject, attribute=attribute, value=value, unit=unit, quote=quote or value),
                      "body.md", section, (1,))


def test_copies_are_grouped_and_the_odd_one_found():
    facts = [_pf("Project", "mine life", "41", "years", "3.3"), _pf("Project", "mine life", "41", "yrs", "6.3"),
             _pf("the Project", "life of mine", "41", "years", "1"), _pf("open pit", "mine life", "45", "years", "3.4"),
             _pf("Phase 1", "duration", "4", "years", "3.3.2")]
    groups = group_facts(facts)
    assert sorted(len(g) for g in groups) == [1, 4]
    [d] = disagreements(groups)
    assert d.majority == (41.0, 41.0) and [d.facts[i].fact.value for i in d.odd] == ["45"]


def test_agreeing_or_single_place_groups_are_not_disagreements():
    same = [_pf("CTFS", "side slope", "5:1", "H:V", "3.17"), _pf("CTFS", "side slope", "5H:1V", "H:V", "6.5.9")]
    one_place = [_pf("CTFS", "area", "1166.1", "acres", "3.1"), _pf("CTFS", "area", "1616.1", "acres", "3.1")]
    assert disagreements(group_facts(same)) == []
    assert disagreements(group_facts(one_place)) == []


def test_generic_subjects_need_an_identical_attribute():
    facts = [_pf("Project", "disturbance", "5545", "acres", "3.1"), _pf("CTFS", "disturbance area", "1166", "acres", "3.17")]
    assert len(group_facts(facts)) == 2


# ----------------------------------------------------------------- chunks and table arithmetic
def test_numeric_chunks_pick_quantities_not_acronyms():
    ix = PlanIndex.load(CASES / "TP-000")
    chunks = numeric_chunks(ix)
    assert 280 <= len(chunks) <= 400
    assert ix.chunk_at("body.md", 90346) in chunks                       # West WRSF capacities
    assert not any("acronym" in c.path.lower() for c in chunks)


def test_table_totals_flag_only_the_planted_mismatch():
    def notes(case):
        ix = PlanIndex.load(CASES / case)
        return [n for c in numeric_chunks(ix) for n in table_total_checks(PAGE_MARKER.sub(" ", ix.chunk_text(c)))]
    assert notes("TP-000") == []
    [note] = notes("TP-029")
    assert "5,995.0" in note and "5,545.0" in note


def test_table_total_rules():
    table = ("| Facility | Acres |\n|---|---|\n| Previous | |\n| Old site | (50.5) [2] |\n"
             "| Pit | 10.0 |\n| Dump | 5.0 |\n| Total | 16.0 |")
    assert table_total_checks(table) == ['Table row "Total": column "Acres" rows sum to 15.0 but the total says 16.0']
    assert table_total_checks(table.replace("16.0", "15.0")) == []


# ----------------------------------------------------------------- the value path, scripted
def _scripted(module, chunk_answer, conflict_answer=None):
    """Replace the two batch calls: chunk_answer(passage) and conflict_answer(statements) -> Prediction."""
    module.check_chunk.batch = lambda exs, **kw: [chunk_answer(e.passage) for e in exs]
    module.check_conflict.batch = lambda exs, **kw: [(conflict_answer or (lambda s: dspy.Prediction(
        conflict=False, wrong=[], checklist_id="", reason="", confidence=0.0)))(e.statements) for e in exs]
    return module


def _caught(case_id, flags):
    ex = load_devset("thacker_pass_clean", case_id)[0]
    case = bench_score.Case.from_dict(ex.gold)
    return any(bench_score.hit(bench_score.Flag.from_dict(f), case) for f in flags)


def test_a_local_finding_becomes_a_caught_flag():
    from agent.systems.v2 import ValueFinding, ValueReviewer

    def chunk_answer(passage):
        findings = []
        if "| West WRSF | 37.2 |" in passage:
            findings = [ValueFinding(status="inconsistent", checklist_id="ops-designs", quote="| West WRSF | 37.2 | 5,252 |",
                                     other_quote="approximately 32.7 M CY", reason="text vs table", confidence=0.9)]
        return dspy.Prediction(facts=[], findings=findings)

    pred = _scripted(ValueReviewer(), chunk_answer)(case_dir=str(CASES / "TP-021"))
    [flag] = pred.flags
    assert (flag["file"], flag["section"], flag["pages"], flag["path"]) == ("body.md", "3.5", [51], "value-local")
    assert _caught("TP-021", pred.flags)


def test_a_distant_conflict_becomes_a_caught_flag():
    from agent.systems.v2 import ValueReviewer

    def chunk_answer(passage):
        facts = []
        if "throughout the 45-year mine life" in passage:
            facts = [Fact(subject="Project", attribute="mine life", value="45", unit="years",
                          quote="throughout the 45-year mine life")]
        elif "over the estimated 41-year mine life" in passage:
            facts = [Fact(subject="Project", attribute="mine life", value="41", unit="years",
                          quote="over the estimated 41-year mine life")]
        return dspy.Prediction(facts=facts, findings=[])

    def conflict_answer(statements):
        wrong = [1] if statements.split("\n\n")[0].count("45") else [2]
        return dspy.Prediction(conflict=True, wrong=wrong, checklist_id="ops-schedule", reason="45 vs 41", confidence=0.8)

    pred = _scripted(ValueReviewer(), chunk_answer, conflict_answer)(case_dir=str(CASES / "TP-025"))
    assert [f["section"] for f in pred.flags] == ["3.4"] and pred.flags[0]["path"] == "value-distant"
    assert _caught("TP-025", pred.flags)
    assert pred.value_log["groups"][0]["conflict"] is True


def test_merge_keeps_one_flag_per_item_and_place():
    from agent.systems.v2 import merge_reports
    f = {"checklist_id": "ops-designs", "type": "inconsistent", "file": "body.md", "section": "3.5", "pages": [51],
         "quote": "x", "explanation": "", "anchor": "quote"}
    a = {"case_id": "TP-021", "flags": [f]}
    b = {"case_id": "TP-021", "flags": [{**f, "type": "vague", "path": "value-local"},
                                        {**f, "checklist_id": "rec-topsoil", "path": "value-local"}]}
    merged = merge_reports("TP-021", "v2", ("checklist", a), ("value", b))
    assert [(x["checklist_id"], x["path"]) for x in merged["flags"]] == [("ops-designs", "checklist"),
                                                                        ("rec-topsoil", "value-local")]


# ----------------------------------------------------------------- fact probe
def test_fact_probe_reports_extraction_and_grouping():
    from harness.fact_probe import probe_case
    gold = load_devset("thacker_pass_clean", "TP-025")[0].gold
    content = next(l for l in gold["locations"] if l["role"] == "content")
    related = next(l for l in gold["locations"] if l["role"] == "related")
    at = lambda l, v: {"subject": "Project", "attribute": "mine life", "value": v, "unit": "years", "quote": v,
                       "file": l["file"], "section": l["section"], "pages": [l["page"]]}
    log = {"facts": [at(content, "45"), at(related, "41")], "local": [],
           "groups": [{"statements": [at(content, "45"), at(related, "41")], "conflict": False}]}
    row = probe_case(gold, log, caught=False)
    assert row["extracted"] and row["grouped"] and row["label"] == "value-judge miss"
    assert probe_case(gold, {"facts": [], "local": [], "groups": []}, caught=False)["label"] == "extraction miss"
