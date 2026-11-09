"""Tests for the DSPy harness (harness/): cases, metric, runner and search check. Offline.

Run:  uv run pytest -q
"""

import json
from pathlib import Path

import dspy

from bench import score as bench_score
from harness.data import SMOKE, load_devset
from harness.metric import bench_metric, search_hit

ROOT = Path(__file__).resolve().parents[1]


def test_examples_give_programs_only_the_case_folder():
    devset = load_devset("thacker_pass_clean", "smoke")
    assert [ex.case_id for ex in devset] == SMOKE
    for ex in devset:
        assert set(ex.inputs().keys()) == {"case_dir"}
        assert "gold" in ex and "gold" not in ex.inputs()


def test_bench_metric_matches_the_scorer_on_perfect_reports():
    key = json.loads((ROOT / "bench" / "answer_key" / "thacker_pass_clean.json").read_text())
    perfect = bench_score.oracle_reports(key)
    for ex in load_devset("thacker_pass_clean", "all"):
        assert bench_metric(ex, dspy.Prediction(flags=perfect[ex.case_id])) == 1.0


def test_bench_metric_charges_false_alarms_and_misses():
    ex = load_devset("thacker_pass_clean", "TP-021")[0]
    alarm = {"checklist_id": "rec-topsoil", "type": "vague", "file": "body.md", "section": "6.4.4", "explanation": ""}
    assert bench_metric(ex, dspy.Prediction(flags=[])) == 0.0
    assert bench_metric(load_devset("thacker_pass_clean", "TP-000")[0], dspy.Prediction(flags=[alarm])) == 0.9


def test_search_hit_uses_the_scorers_place_rule():
    from agent.plan import Location
    ex = load_devset("thacker_pass_clean", "TP-021")[0]
    assert search_hit([Location("body.md", "6.4"), Location("body.md", "3.5", (51,))], ex.gold) == 2
    assert search_hit([Location("appendix_C.md", "3.5")], ex.gold) is None


def test_replay_run_catches_tp021_end_to_end(tmp_path):
    """case -> index -> locator -> report -> runs/ -> official scorer: caught, no false alarms."""
    from harness.run import run
    summary = run("replay", cases="TP-021", name="t", runs_root=tmp_path, threads=1, quiet=True)
    assert summary["caught"] == 1 and summary["false_alarms"] == 0 and summary["errors"] == 0
    report = json.loads((tmp_path / "t" / "TP-021.json").read_text())
    assert report["flags"][0]["anchor"] == "quote"
    key = json.loads((ROOT / "bench" / "answer_key" / "thacker_pass_clean.json").read_text())
    res = bench_score.score(key, bench_score.load_reports(tmp_path / "t", ["TP-021"]))
    row = next(r for r in res["cases"] if r["case_id"] == "TP-021")
    assert row["caught"] and row["false_alarms"] == 0


def test_runner_survives_a_crashing_program(tmp_path, monkeypatch):
    from harness import run as run_mod

    class Boom(dspy.Module):
        def forward(self, case_dir):
            raise RuntimeError("boom")

    monkeypatch.setattr(run_mod, "systems", lambda: {"boom": Boom})
    summary = run_mod.run("boom", cases="TP-000,TP-021", name="b", runs_root=tmp_path, threads=1, quiet=True)
    assert summary["errors"] == 2
    assert json.loads((tmp_path / "b" / "TP-000.json").read_text())["flags"] == []


def test_search_check_runs_and_finds_most_flaws():
    from harness.probe import per_case, probe
    cases = per_case(probe(ks=(5,)))
    assert len(cases) == 37                       # 40 flawed cases minus 3 whole-file deletions
    found = sum(c["rank"] is not None and c["rank"] <= 5 for c in cases)
    assert found / len(cases) >= 0.6              # V0 baseline was 0.73; guards against regressions


def test_false_alarm_counts_split_standing_from_case_specific():
    from harness.misses import false_alarm_counts
    k1, k2 = ("ops-maps", "body.md", "3.1"), ("rec-topsoil", "body.md", "6.4")
    keys = [("TP-000", k1), ("TP-001", k1), ("TP-002", k1), ("TP-002", k2)]
    assert false_alarm_counts(keys, {k1}) == {"total": 4, "control": 1, "standing": 2, "case_specific": 1, "distinct": 2}
    assert false_alarm_counts(keys, None)["standing"] == 0
