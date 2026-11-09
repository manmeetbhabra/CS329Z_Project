"""Test cases as dspy.Examples.

    devset = load_devset("thacker_pass_clean", "smoke")
    ex = devset[0]
    ex.inputs()       # {"case_dir": "bench/cases/thacker_pass_clean/TP-000"}  <- all a program sees
    ex.gold           # the answer-key entry, for the metric only
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = "thacker_pass_clean"

# TP-000 (control) + 2 per flaw type, mixed difficulty.
SMOKE = ["TP-000",
         "TP-002", "TP-006",      # missing: easy, subtle
         "TP-011", "TP-017",      # vague: easy, medium
         "TP-021", "TP-029",      # inconsistent: easy, subtle
         "TP-035", "TP-039"]      # implausible: medium, subtle


def answer_key(plan: str = DEFAULT_PLAN) -> dict:
    return json.loads((ROOT / "bench" / "answer_key" / f"{plan}.json").read_text(encoding="utf-8"))


def select(case_ids: list[str], cases: str | list[str] = "all") -> list[str]:
    """'all' | 'smoke' | 'flawed' | 'TP-000,TP-021' | a list of ids."""
    if isinstance(cases, list):
        wanted = cases
    elif cases == "all":
        return list(case_ids)
    elif cases == "smoke":
        wanted = SMOKE
    elif cases == "flawed":
        return [c for c in case_ids if c != "TP-000"]
    else:
        wanted = [c.strip() for c in cases.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in case_ids]
    if unknown:
        raise ValueError(f"unknown case ids: {unknown}")
    return [c for c in case_ids if c in wanted]


def load_devset(plan: str = DEFAULT_PLAN, cases: str | list[str] = "all") -> list[dspy.Example]:
    """One dspy.Example per case. Input: case_dir (relative to the project root). The answer-key
    entry rides along as `gold` for the metric; programs never receive it."""
    key = answer_key(plan)
    by_id = {c["case_id"]: c for c in key["cases"]}
    return [dspy.Example(case_id=cid, case_dir=f"bench/cases/{plan}/{cid}", gold=by_id[cid]).with_inputs("case_dir")
            for cid in select(list(by_id), cases)]
