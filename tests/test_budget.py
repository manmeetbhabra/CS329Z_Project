"""Tests for the project spending limit (utils/budget.py) and the budgeted LM (agent/config.py).
Offline: OpenRouter's usage figure is faked.

Run:  uv run pytest -q
"""

import pytest

from utils import budget


@pytest.fixture
def fake_usage(monkeypatch):
    """Budget $10 from a starting usage of $1.00; `fake_usage["now"]` is what OpenRouter reports."""
    monkeypatch.setenv("PROJECT_BUDGET_USD", "10")
    monkeypatch.setenv("PROJECT_BUDGET_START_USAGE", "1.00")
    state = {"now": 1.00, "asked": 0}

    def usage_now():
        state["asked"] += 1
        return state["now"]

    monkeypatch.setattr(budget, "usage_now", usage_now)
    return state


def test_check_returns_whats_left(fake_usage):
    fake_usage["now"] = 1.42
    assert budget.check() == pytest.approx(9.58)


def test_check_stops_when_the_budget_is_spent(fake_usage):
    fake_usage["now"] = 11.01
    with pytest.raises(budget.BudgetExceeded):
        budget.check()


def test_no_budget_set_means_no_limit(monkeypatch):
    monkeypatch.setenv("PROJECT_BUDGET_USD", "")
    monkeypatch.setattr(budget, "load_dotenv", lambda *a, **k: None)
    assert budget.check() is None


def test_unreachable_openrouter_fails_closed(fake_usage, monkeypatch):
    def boom():
        raise OSError("offline")
    monkeypatch.setattr(budget, "usage_now", boom)
    with pytest.raises(RuntimeError, match="Can't check the project budget"):
        budget.check()


def test_meter_checks_every_n_calls_and_then_blocks_for_good(fake_usage):
    meter = budget.Meter(check_every=3)
    for _ in range(3):
        meter.tick()                      # call 1 checks; calls 2-3 don't
    assert fake_usage["asked"] == 1
    fake_usage["now"] = 20.0              # the budget runs out
    with pytest.raises(budget.BudgetExceeded):
        meter.tick()                      # call 4 checks and trips
    asked = fake_usage["asked"]
    with pytest.raises(budget.BudgetExceeded):
        meter.tick()                      # blocked at once ...
    assert fake_usage["asked"] == asked   # ... without asking OpenRouter again


def test_budgeted_lm_refuses_before_any_network_call(fake_usage, monkeypatch):
    from agent import config
    meter = budget.Meter(check_every=1)
    monkeypatch.setattr(config, "METER", meter)
    sent = []
    monkeypatch.setattr(config.dspy.LM, "__call__", lambda self, *a, **kw: sent.append(kw) or ["ok"])
    lm = config.BudgetedLM("openrouter/openai/gpt-6-luna")
    assert lm(messages=[{"role": "user", "content": "hi"}]) == ["ok"] and len(sent) == 1
    fake_usage["now"] = 20.0
    with pytest.raises(budget.BudgetExceeded):
        lm(messages=[{"role": "user", "content": "hi again"}])
    assert len(sent) == 1                 # the second call never reached the provider


def test_runner_refuses_to_start_a_paid_run_over_budget(fake_usage, monkeypatch, tmp_path):
    import dspy
    from harness import run as run_mod

    class Paid(dspy.Module):              # has a predictor, so the runner treats it as a paid system
        def __init__(self):
            super().__init__()
            self.p = dspy.Predict("question -> answer")

        def forward(self, case_dir):
            raise AssertionError("must not run")

    monkeypatch.setattr(run_mod, "systems", lambda: {"paid": Paid})
    fake_usage["now"] = 11.5
    with pytest.raises(budget.BudgetExceeded):
        run_mod.run("paid", cases="TP-000", name="x", runs_root=tmp_path, quiet=True)
    assert not (tmp_path / "x" / "TP-000.json").exists()
