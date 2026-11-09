"""Replay: a fake reviewer that returns hand-written findings, to test the whole path with no AI.

For case TP-021 it reads tests/fixtures/replay_TP-021.json, a list of FlagDraft dicts that give
quotes only (no places). The flags then go through the locator and the report writer exactly as
a real reviewer's would, so a catch on TP-021 proves: case -> index -> locator -> report -> score.
Cases without a fixture get an empty report.
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy

from agent.anchor import anchor
from agent.plan import PlanIndex
from agent.report import FlagDraft, to_report

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


class Replay(dspy.Module):
    name = "replay"

    def __init__(self, fixtures: Path = FIXTURES):
        super().__init__()
        self.fixtures = Path(fixtures)

    def forward(self, case_dir: str) -> dspy.Prediction:
        index = PlanIndex.load(case_dir)
        path = self.fixtures / f"replay_{index.name}.json"
        drafts = [FlagDraft(**d) for d in json.loads(path.read_text())] if path.exists() else []
        report = to_report(index.name, self.name, [anchor(d, index) for d in drafts])
        return dspy.Prediction(flags=report["flags"], report=report)
