"""The project's spending limit, enforced in our code.

    uv run python -m utils.budget          # limit, spent so far, remaining

Set in .env:
    PROJECT_BUDGET_USD=10                  # how much this project may spend
    PROJECT_BUDGET_START_USAGE=1.0023      # the key's total usage when the budget was set

"Spent" = OpenRouter's usage for the key now - PROJECT_BUDGET_START_USAGE, so it counts every
call made with the key (runs, notebooks, anything). To start a fresh budget, set the start to the
current usage (`python -m utils.balance` shows it).

Who enforces it:
  check()        raises BudgetExceeded when spent >= limit. harness.run calls it before a paid run.
  Meter.tick()   called by agent.config.BudgetedLM before every LLM call; asks OpenRouter every
                 CHECK_EVERY calls, and once the budget is exceeded blocks every later call at once.
If OpenRouter can't be reached, check() fails closed (raises) rather than spending blind.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from utils.balance import usage_now

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK_EVERY = 25          # LLM calls between checks with OpenRouter (usage there can lag a few seconds)


class BudgetExceeded(RuntimeError):
    """The project budget is used up; no more paid calls are made."""


@dataclass(frozen=True)
class Budget:
    limit: float          # dollars this project may spend, e.g. 10.0
    start: float          # the key's usage when the budget was set, e.g. 1.0023


def load_budget() -> Budget | None:
    """The budget from .env, or None if PROJECT_BUDGET_USD isn't set (no project limit).
    PROJECT_BUDGET_USD=10, PROJECT_BUDGET_START_USAGE=1.0023 -> Budget(limit=10.0, start=1.0023)"""
    load_dotenv(REPO_ROOT / ".env")
    limit = os.environ.get("PROJECT_BUDGET_USD")
    if not limit:
        return None
    return Budget(float(limit), float(os.environ.get("PROJECT_BUDGET_START_USAGE") or 0.0))


def status(budget: Budget | None = None) -> dict:
    """{"limit": 10.0, "spent": 0.42, "remaining": 9.58}; asks OpenRouter for the current usage.
    With no budget set: {"limit": None, "spent": None, "remaining": None}."""
    budget = budget or load_budget()
    if budget is None:
        return {"limit": None, "spent": None, "remaining": None}
    try:
        spent = usage_now() - budget.start
    except Exception as exc:  # noqa: BLE001 - fail closed: never spend without knowing
        raise RuntimeError(f"Can't check the project budget (OpenRouter usage unavailable: {exc})") from exc
    return {"limit": budget.limit, "spent": round(spent, 4), "remaining": round(budget.limit - spent, 4)}


def check(budget: Budget | None = None) -> float | None:
    """Raise BudgetExceeded if the budget is spent; otherwise return what's left (None if no budget).
    spent $0.42 of $10 -> 9.58;   spent $10.03 of $10 -> BudgetExceeded"""
    s = status(budget)
    if s["limit"] is not None and s["remaining"] <= 0:
        raise BudgetExceeded(f"Project budget used up: spent ${s['spent']:.2f} of ${s['limit']:.2f} "
                             "(PROJECT_BUDGET_USD in .env). Raise it there to continue.")
    return s["remaining"]


class Meter:
    """Counts LLM calls and checks the budget every CHECK_EVERY calls (and on the first one).
    Once exceeded it stays exceeded, so later calls fail at once without asking OpenRouter."""

    def __init__(self, check_every: int = CHECK_EVERY):
        self.check_every = check_every
        self.calls = 0
        self.exceeded: BudgetExceeded | None = None
        self._lock = threading.Lock()

    def tick(self) -> None:
        with self._lock:
            if self.exceeded is not None:
                raise self.exceeded
            due = self.calls % self.check_every == 0
            self.calls += 1
            if due:
                try:
                    check()
                except BudgetExceeded as exc:
                    self.exceeded = exc
                    raise


METER = Meter()          # one per process, shared by every BudgetedLM


def report() -> str:
    budget = load_budget()
    if budget is None:
        return "No project budget set (add PROJECT_BUDGET_USD to .env)."
    s = status(budget)
    return (f"  limit:     ${s['limit']:.2f}  (PROJECT_BUDGET_USD)\n"
            f"  spent:     ${s['spent']:.4f}  (since key usage ${budget.start:.4f})\n"
            f"  remaining: ${s['remaining']:.4f}")


if __name__ == "__main__":
    print(report())
