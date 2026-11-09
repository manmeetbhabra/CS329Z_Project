"""Findings and reports.

    FlagDraft      one finding as a reviewer (the V1 judge, or a test fixture) states it: a checklist
                   item, a type, an exact quote and an explanation. Place fields are optional guesses.
    to_report()    placed flags -> the JSON report the scorer reads (bench/schema/report.schema.json),
                   validated before it is returned, so a misspelled field fails loudly instead of
                   silently losing credit.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import jsonschema
from pydantic import BaseModel, Field

from agent.checklist import items_by_id

if TYPE_CHECKING:
    from agent.anchor import PlacedFlag

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "bench" / "schema" / "report.schema.json"

FlagType = Literal["missing", "vague", "inconsistent", "implausible", "missed_trigger"]


class FlagDraft(BaseModel):
    """One finding before it is placed. The quote is what the locator uses; file, section and
    pages are only the reviewer's guesses (used when there is nothing to quote)."""
    checklist_id: str
    type: FlagType
    quote: str | None = None
    file: str | None = None
    section: str | None = None
    pages: list[int] = Field(default_factory=list)
    explanation: str = ""
    confidence: float | None = None


@lru_cache(maxsize=1)
def _schema() -> dict:
    return json.loads(SCHEMA.read_text(encoding="utf-8"))


def validate(report: dict) -> None:
    """Raise jsonschema.ValidationError if the report doesn't match the scorer's schema."""
    jsonschema.validate(report, _schema())


def to_report(case_id: str, system: str, placed: list["PlacedFlag"]) -> dict:
    """The report for one case. Unknown checklist ids become "other" (the original is kept as
    raw_checklist_id); exact duplicates (same item, type and place) are dropped."""
    known = items_by_id()
    flags, seen = [], set()
    for p in placed:
        d, loc = p.draft, p.location
        file, section, pages = (loc.file, loc.section, list(loc.pages)) if loc else (None, None, [])
        key = (d.checklist_id, d.type, file, section, tuple(pages))
        if key in seen:
            continue
        seen.add(key)
        flag = {"checklist_id": d.checklist_id if d.checklist_id in known else "other",
                "type": d.type, "file": file, "section": section, "pages": pages,
                "quote": d.quote, "explanation": d.explanation or "", "anchor": p.how}
        if d.checklist_id not in known:
            flag["raw_checklist_id"] = d.checklist_id
        if d.confidence is not None:
            flag["confidence"] = max(0.0, min(1.0, float(d.confidence)))
        if p.model_location is not None:
            m = p.model_location
            flag["model_location"] = {"file": m.file, "section": m.section, "pages": list(m.pages)}
        flags.append(flag)
    report = {"case_id": case_id, "system": system, "flags": flags}
    validate(report)
    return report
