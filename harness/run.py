"""Run a reviewer over the bench cases, save its reports, and (optionally) score them.

    uv run python -m harness.run replay --cases TP-021 --name replay-check --score
    uv run python -m harness.run replay --cases smoke  --name replay-smoke

Writes runs/<name>/:
    TP-###.json      one report per case (the format bench/score.py reads)
    run_log.jsonl    one row per case: metric score, caught, false alarms, flags by anchor rung, seconds, error
    run.json         the run: system, plan, cases, DSPy version, mean metric, errors, cost
and with --score, the official scores.json / scores.md from `python -m bench.score`, run as a
separate process (so the official numbers never depend on this code).

dspy.Evaluate runs the program on every case in parallel and grades each with bench_metric.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import dspy

from harness.data import DEFAULT_PLAN, load_devset
from harness.metric import bench_metric, score_row

ROOT = Path(__file__).resolve().parents[1]


def systems() -> dict[str, type]:
    from agent.systems.replay import Replay
    from functools import partial
    from agent.systems.v1 import PlanReviewer
    from agent.systems.v2 import PlanReviewerV2, ValueReviewer
    return {"replay": Replay, "v1": PlanReviewer,
            "v1-map": partial(PlanReviewer, use_map=True),        # ablation: with the whole-plan map
            "v2": PlanReviewerV2,                                 # checklist path + value path
            "v2-values": ValueReviewer}                           # ablation: value path only


class Guarded(dspy.Module):
    """Runs a program on one case and turns any exception into an empty report plus an error,
    so one bad case never stops a run."""

    def __init__(self, program: dspy.Module, system: str):
        super().__init__()
        self.program, self.system = program, system

    def forward(self, case_dir: str) -> dspy.Prediction:
        start = time.time()
        verdicts = value_log = None
        try:
            pred = self.program(case_dir=case_dir)
            error = None
            report = pred.report
            verdicts = getattr(pred, "verdicts", None)
            value_log = getattr(pred, "value_log", None)
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            error = f"{type(exc).__name__}: {exc}"
            report = {"case_id": Path(case_dir).name, "system": self.system, "flags": []}
        return dspy.Prediction(flags=report["flags"], report=report, verdicts=verdicts, value_log=value_log,
                               error=error, seconds=time.time() - start)


def run(system: str, plan: str = DEFAULT_PLAN, cases: str = "all", name: str | None = None,
        threads: int | None = None, runs_root: Path = ROOT / "runs", score: bool = False, quiet: bool = False,
        lm: dspy.LM | None = None, model: str = "dev") -> dict:
    """Run one system over the cases. A system with an LLM (predictors) gets the project budget
    check, the dev model (or `lm`, e.g. a DummyLM in tests), 4 cases at a time by default, saved
    verdicts and token counts. Returns the run summary (also written to runs/<name>/run.json)."""
    program = systems()[system]()
    uses_lm = bool(program.predictors())
    devset = load_devset(plan, cases)
    out = Path(runs_root) / (name or f"{system}-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    threads = threads or (4 if uses_lm else 8)    # an LLM system already runs 8 calls per case in parallel

    budget_left = None
    if uses_lm:
        if lm is None:
            from utils.budget import check
            budget_left = check()          # BudgetExceeded here means nothing is spent; raise PROJECT_BUDGET_USD to go on
            if not quiet and budget_left is not None:
                print(f"Project budget: ${budget_left:.2f} left")
            from agent.config import configure
            lm = configure(model)
        else:
            dspy.configure(lm=lm, adapter=dspy.JSONAdapter(), track_usage=True)
    cost_start = _usage_now() if uses_lm and budget_left is not None else None
    history_start = len(lm.history) if uses_lm else 0
    started = time.time()
    guarded = Guarded(program, system)
    if uses_lm and len(devset) > 1:
        # Warm DSPy's cache with one case first: cases share almost every chunk, so the parallel run
        # that follows reads them from the cache instead of paying for the same calls at once.
        guarded(case_dir=devset[0].case_dir)
    evaluate = dspy.Evaluate(devset=devset, metric=bench_metric, num_threads=threads,
                             display_progress=not quiet, max_errors=len(devset) + 1)
    result = evaluate(guarded)

    rows, errors = [], 0
    with open(out / "run_log.jsonl", "w", encoding="utf-8") as log:
        for ex, pred, metric in result.results:
            report = pred.report
            (out / f"{ex.case_id}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
            if getattr(pred, "verdicts", None) is not None:
                (out / "verdicts").mkdir(exist_ok=True)
                (out / "verdicts" / f"{ex.case_id}.jsonl").write_text(
                    "".join(json.dumps(v) + "\n" for v in pred.verdicts), encoding="utf-8")
            if getattr(pred, "value_log", None) is not None:
                (out / "values").mkdir(exist_ok=True)
                (out / "values" / f"{ex.case_id}.json").write_text(json.dumps(pred.value_log, indent=1), encoding="utf-8")
            r = score_row(ex, pred)
            row = {"case_id": ex.case_id, "type": r["type"], "difficulty": r["difficulty"], "metric": metric,
                   "caught": r["caught"], "n_flags": r["n_flags"], "false_alarms": r["false_alarms"],
                   "anchors": dict(Counter(f.get("anchor", "?") for f in report["flags"])),
                   "seconds": round(getattr(pred, "seconds", 0.0), 2), "error": getattr(pred, "error", None)}
            errors += row["error"] is not None
            rows.append(row)
            log.write(json.dumps(row) + "\n")

    summary = {"system": system, "model": (getattr(lm, "model", None) if uses_lm else None),
               "plan": plan, "cases": cases, "n_cases": len(devset),
               "dspy_version": dspy.__version__, "mean_metric": round(result.score / 100, 4),
               "caught": sum(r["caught"] for r in rows), "flags": sum(r["n_flags"] for r in rows),
               "false_alarms": sum(r["false_alarms"] for r in rows), "errors": errors,
               "seconds": round(time.time() - started, 1),
               "lm_calls": len(lm.history) - history_start if uses_lm else 0,
               "tokens": _tokens(lm.history[history_start:]) if uses_lm else {},
               "cost_usd": _spent_since(cost_start) if cost_start is not None else 0.0,
               "budget_left_before_usd": budget_left,
               "params": {k: getattr(program, k) for k in ("k", "min_confidence", "passage_chars", "use_map")
                          if hasattr(program, k)},
               "runs_dir": str(out)}
    if any((r["error"] or "").startswith("BudgetExceeded") for r in rows):
        print("WARNING: the project budget ran out during this run; later cases have empty reports. "
              "Raise PROJECT_BUDGET_USD in .env and re-run (finished cases come from DSPy's cache).")
    (out / "run.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if not quiet:
        print(json.dumps(summary, indent=1))
    if score:
        subprocess.run([sys.executable, "-m", "bench.score", plan, str(out)], cwd=ROOT, check=True)
    return summary


def _tokens(history: list[dict]) -> dict:
    """Prompt and completion tokens reported in these LM history entries (one entry per call;
    replies served from DSPy's cache report their original usage). -> {"prompt": 210345, "completion": 31020}"""
    prompt = completion = 0
    for h in history:
        u = h.get("usage") or {}
        prompt += int(u.get("prompt_tokens") or 0)
        completion += int(u.get("completion_tokens") or 0)
    return {"prompt": prompt, "completion": completion}


def _usage_now() -> float | None:
    try:
        from utils.balance import usage_now
        return usage_now()
    except Exception:  # noqa: BLE001 - cost is informational
        return None


def _spent_since(start: float | None) -> float | None:
    if start is None:
        return None
    now = _usage_now()
    return None if now is None else round(now - start, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system", choices=sorted(systems()))
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--cases", default="all", help="all | smoke | flawed | TP-000,TP-021")
    ap.add_argument("--name", help="run folder name under runs/ (default: <system>-<timestamp>)")
    ap.add_argument("--threads", type=int, default=None, help="cases at a time (default 8, or 4 for an LLM system)")
    ap.add_argument("--score", action="store_true", help="also run the official scorer on the reports")
    ap.add_argument("--model", default="dev", help="model alias from agent/config.py MODELS: dev (gpt-6-luna) | strong (gpt-6-sol)")
    args = ap.parse_args()
    run(args.system, args.plan, args.cases, args.name, args.threads, score=args.score, model=args.model)


if __name__ == "__main__":
    main()
