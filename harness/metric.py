"""The bench's own rule, as DSPy metrics.

    bench_metric(example, pred)          one case's score for dspy.Evaluate (and later, optimizers)
    search_hit(locations, gold)          does any searched place sit at the planted flaw?

Both call bench/score.py, so DSPy grades with exactly the rule the official scorer uses.
"""

from __future__ import annotations

from bench import score as bench_score

FALSE_ALARM_COST = 0.1


def bench_metric(example, pred, trace=None) -> float:
    """1 if the planted flaw is caught (0 if not), minus FALSE_ALARM_COST per false alarm, floored
    at 0. On the clean control, 1 minus the same cost per flag. Duplicates are free, as in the scorer."""
    flags = list(getattr(pred, "flags", None) or [])
    row = bench_score.score_case(example.gold, flags)
    base = 1.0 if (row["caught"] or not row["planted"]) else 0.0
    return max(0.0, base - FALSE_ALARM_COST * row["false_alarms"])


def score_row(example, pred) -> dict:
    """The scorer's full row for one case (caught, false alarms, ...), for run logs."""
    return bench_score.score_case(example.gold, list(getattr(pred, "flags", None) or []))


def to_bench_location(loc) -> bench_score.Location:
    """agent.plan.Location -> bench.score.Location (same fields)."""
    return bench_score.Location(loc.file, loc.section, tuple(loc.pages))


def gold_case(gold: dict) -> bench_score.Case:
    return bench_score.Case.from_dict(gold)


def search_hit(locations, gold: dict) -> int | None:
    """Rank (1-based) of the first location at a planted-flaw place, or None. Uses the scorer's
    same_place(): same file, and same section or subsection, or pages within +/-1."""
    case = gold_case(gold)
    for rank, loc in enumerate(locations, 1):
        b = to_bench_location(loc)
        if any(bench_score.same_place(b, g) for g in case.locations):
            return rank
    return None
