"""V2: V1's checklist path plus a value path that reads every number-bearing chunk once.

    from agent.config import configure
    from agent.systems.v2 import PlanReviewerV2, ValueReviewer
    configure("dev")
    pred = PlanReviewerV2()(case_dir="bench/cases/thacker_pass_clean/TP-021")   # both paths, merged
    pred = ValueReviewer()(case_dir=...)                                         # value path only
    pred.value_log   # facts, disagreeing groups, local and distant findings (for harness/fact_probe.py)

The value path:
  A  CheckChunk     per number-bearing chunk: list its facts; report local conflicts, totals that
                    don't add up, and implausible values (each tagged with a checklist item)
  B  fact ledger    plain code (agent/facts.py): group copies of the same fact across the plan,
                    keep the groups whose values disagree
  C  CheckConflict  per disagreeing group: a real conflict? which copies are wrong? which item?
Identical chunks and groups across cases hit DSPy's disk cache, so a bench run pays for roughly one
plan's worth of reading plus the few chunks each planted edit changes.
"""

from __future__ import annotations

from typing import Literal

import dspy
from pydantic import BaseModel

from agent.anchor import anchor
from agent.checklist import render_catalog
from agent.facts import Fact, Group, PlacedFact, disagreements, group_facts, numeric_chunks, table_total_checks
from agent.plan import PAGE_MARKER, Chunk, PlanIndex
from agent.report import FlagDraft, to_report
from agent.systems.v1 import PlanReviewer

CHUNK_CHARS = 4000          # a chunk longer than this is cut (chunks are <= ~5,500 characters)
VALUE_MAX_TOKENS = 8000     # a numeric table can need long answers; 4,000 cut 12 of 153 replies short
MAX_GROUPS = 80             # disagreeing groups judged per case, most places first


class ValueFinding(BaseModel):
    """One problem with a value in a chunk."""
    status: Literal["inconsistent", "implausible"]
    checklist_id: str           # one id from the catalog
    quote: str                  # the offending value, copied exactly
    other_quote: str            # the value it conflicts with (inconsistent), else ""
    reason: str
    confidence: float           # 0 to 1


class CheckChunk(dspy.Signature):
    """You check the numbers in ONE passage of a Plan of Operations for a mine on federal land
    (BLM, 43 CFR 3809).
    1. List the quantities the passage states as facts (at most 30; if there are more, keep the
       design and operating values - sizes, capacities, rates, slopes, dates, durations, totals):
       subject, attribute, value (as written), unit, and the exact words containing it. Include table
       cells (subject = row, attribute = column). Skip page numbers, section numbers, citations,
       dates of documents and phone numbers.
    2. Report a finding only if:
       (a) two values in this passage disagree about the same thing under the same conditions
           (e.g. the text and a table, or two items of one list);
       (b) parts stated in this passage do not add up to a total it states (the 'arithmetic' input
           lists table totals that code already found not to add up - check and report them); or
       (c) a design or performance value is one no mining engineer or BLM reviewer would accept
           for what it describes.
       Different phases, facilities, scenarios or units are not conflicts. Rounding is not a
       conflict. No findings is the normal answer.
    Tag each finding with the ONE checklist item it bears on. Copy values exactly into quotes."""

    passage: str = dspy.InputField(desc="'file · §section · pages · heading', then the passage text")
    arithmetic: str = dspy.InputField(desc="table totals that code found not to add up in this passage, or 'none'")
    catalog: str = dspy.InputField(desc="checklist items, one per line: 'id: requirement'")
    facts: list[Fact] = dspy.OutputField()
    findings: list[ValueFinding] = dspy.OutputField(desc="usually empty")


class CheckConflict(dspy.Signature):
    """These quoted statements from ONE mine Plan of Operations were grouped by code as possibly
    stating the same fact with different values. Decide whether they truly conflict - the same
    thing, under the same conditions - or describe different things (different facilities, phases,
    periods, scenarios, or a part versus a total). If they conflict, list the statements most likely
    wrong (usually the odd one out) and the ONE checklist item the conflict bears on."""

    statements: str = dspy.InputField(desc="numbered quotes, each headed 'file · §section · pages'")
    catalog: str = dspy.InputField(desc="checklist items, one per line: 'id: requirement'")
    conflict: bool = dspy.OutputField()
    wrong: list[int] = dspy.OutputField(desc="numbers of the statements most likely wrong; empty if no conflict")
    checklist_id: str = dspy.OutputField(desc="one id from the catalog; empty if no conflict")
    reason: str = dspy.OutputField(desc="one or two sentences")
    confidence: float = dspy.OutputField(desc="0 to 1")


def render_chunk(c: Chunk, index: PlanIndex, max_chars: int = CHUNK_CHARS) -> str:
    """'body.md · §3.5 · p.51-52 · 3.5 Waste Rock Storage Facilities\\n<text without page markers>'"""
    pages = f"p.{c.pages[0]}" + (f"-{c.pages[-1]}" if len(c.pages) > 1 else "") if c.pages else "no page"
    head = " · ".join([c.file, f"§{c.section}" if c.section else "no section", pages] + ([c.heading] if c.heading else []))
    text = PAGE_MARKER.sub("", index.chunk_text(c)).strip()
    return head + "\n" + (text if len(text) <= max_chars else text[:max_chars] + " […]")


def render_group(g: Group) -> str:
    """'[1] body.md · §3.3.3 · p.45\\n"41-year mine life"\\n\\n[2] body.md · §3.4 · p.47\\n"45-year mine life"'"""
    lines = []
    for i, pf in enumerate(g.facts, 1):
        pages = ",".join(map(str, pf.pages)) or "-"
        lines.append(f"[{i}] {pf.file} · §{pf.section or '-'} · p.{pages}\n"
                     f"{pf.fact.subject} — {pf.fact.attribute}: {pf.fact.value} {pf.fact.unit}\n\"{pf.fact.quote}\"")
    return "\n\n".join(lines)


def _arithmetic(c: Chunk, index: PlanIndex) -> str:
    """Code's table-total notes for this chunk, or 'none' (agent.facts.table_total_checks)."""
    notes = table_total_checks(PAGE_MARKER.sub(" ", index.chunk_text(c)))
    return "\n".join(notes) if notes else "none"


def _conf(x) -> float:
    try:
        return max(0.0, min(1.0, float(x or 0.0)))
    except (TypeError, ValueError):
        return 0.0


class ValueReviewer(dspy.Module):
    """The value path. forward(case_dir) -> dspy.Prediction(flags, report, value_log)."""

    def _use_long_answers(self) -> None:
        """Give this path's two predictors a copy of the configured LM with VALUE_MAX_TOKENS, without
        changing the shared LM (its settings are part of every cached V1 answer's key)."""
        base = dspy.settings.lm
        if base is None or not hasattr(base, "copy") or (getattr(base, "kwargs", {}) or {}).get("max_tokens", 0) >= VALUE_MAX_TOKENS:
            return
        if getattr(self, "_long_lm_for", None) is not base:
            self._long_lm, self._long_lm_for = base.copy(max_tokens=VALUE_MAX_TOKENS), base
        for p in (self.check_chunk, self.check_conflict):
            p.set_lm(self._long_lm)

    name = "v2-values"

    def __init__(self, threads: int = 8, min_confidence: float = 0.5):
        super().__init__()
        self.check_chunk = dspy.ChainOfThought(CheckChunk)
        self.check_conflict = dspy.ChainOfThought(CheckConflict)
        self.threads, self.min_confidence = threads, min_confidence

    def forward(self, case_dir: str) -> dspy.Prediction:
        self._use_long_answers()
        index = PlanIndex.load(case_dir)
        catalog = render_catalog()
        chunks = numeric_chunks(index)                                                   # no AI
        reads = self.check_chunk.batch(
            [dspy.Example(passage=render_chunk(c, index), arithmetic=_arithmetic(c, index), catalog=catalog
                          ).with_inputs("passage", "arithmetic", "catalog")
             for c in chunks], num_threads=self.threads, disable_progress_bar=True, max_errors=len(chunks))

        facts, local = [], []
        for c, r in zip(chunks, reads):
            if r is None:
                continue
            for f in getattr(r, "facts", None) or []:
                fact = f if isinstance(f, Fact) else Fact(**dict(f))
                facts.append(PlacedFact(fact, c.file, c.section, c.pages, c.id))
            for f in getattr(r, "findings", None) or []:
                d = f.model_dump() if hasattr(f, "model_dump") else dict(f)
                local.append({**d, "confidence": _conf(d.get("confidence")), "file": c.file, "section": c.section,
                              "chunk_id": c.id, "path": "value-local"})

        groups = sorted(disagreements(group_facts(facts)), key=lambda g: -len({pf.place for pf in g.facts}))[:MAX_GROUPS]
        judged = self.check_conflict.batch(
            [dspy.Example(statements=render_group(g), catalog=catalog).with_inputs("statements", "catalog")
             for g in groups], num_threads=self.threads, disable_progress_bar=True, max_errors=len(groups) + 1) if groups else []
        distant, group_log = [], []
        for g, j in zip(groups, judged):
            row = {"key": g.key, "statements": [pf.as_dict() for pf in g.facts], "odd": g.odd,
                   "conflict": bool(getattr(j, "conflict", False)) if j is not None else None,
                   "wrong": list(getattr(j, "wrong", []) or []) if j is not None else [],
                   "checklist_id": (getattr(j, "checklist_id", "") or "").strip() if j is not None else "",
                   "reason": (getattr(j, "reason", "") or "").strip() if j is not None else "",
                   "confidence": _conf(getattr(j, "confidence", 0.0)) if j is not None else 0.0}
            group_log.append(row)
            if row["conflict"] and row["checklist_id"]:
                for n in row["wrong"] or [i + 1 for i in g.odd]:
                    if 1 <= n <= len(g.facts):
                        pf = g.facts[n - 1]
                        others = [g.facts[i].fact.quote for i in range(len(g.facts)) if i != n - 1][:2]
                        distant.append({"status": "inconsistent", "checklist_id": row["checklist_id"],
                                        "quote": pf.fact.quote, "other_quote": " / ".join(others),
                                        "reason": row["reason"], "confidence": row["confidence"],
                                        "file": pf.file, "section": pf.section, "chunk_id": pf.chunk_id,
                                        "path": "value-distant"})

        placed, kept, seen = [], [], set()
        for f in local + distant:
            if f["confidence"] < self.min_confidence or not f.get("checklist_id"):
                continue
            explanation = f["reason"] + (f" (conflicts with: {f['other_quote']})" if f.get("other_quote") else "")
            draft = FlagDraft(checklist_id=f["checklist_id"], type=f["status"], quote=f["quote"] or None,
                              file=f["file"], section=f["section"], explanation=explanation, confidence=f["confidence"])
            p = anchor(draft, index)
            loc = p.location
            f["placed"] = {"how": p.how, "file": loc.file if loc else None,
                           "section": loc.section if loc else None, "pages": list(loc.pages) if loc else []}
            key = (draft.checklist_id, draft.type, loc.file if loc else None, loc.section if loc else None,
                   tuple(loc.pages) if loc else ())
            if key in seen:                                  # same flag twice: keep the first
                continue
            seen.add(key)
            placed.append(p)
            kept.append(f)
        report = to_report(index.name, self.name, placed)    # no exact repeats left, so flags align with kept
        for flag, f in zip(report["flags"], kept):
            flag["path"] = f["path"]
        log = {"n_chunks": len(chunks), "facts": [pf.as_dict() for pf in facts], "groups": group_log,
               "local": local, "distant": distant}
        return dspy.Prediction(flags=report["flags"], report=report, value_log=log)


def merge_reports(case_id: str, system: str, *reports: tuple[str, dict]) -> dict:
    """One report from several paths: each flag tagged with its path; a flag with the same checklist
    item, file and section as an earlier one is dropped. merge_reports("TP-021", "v2", ("checklist", r1), ("value", r2))"""
    flags, seen = [], set()
    for path, report in reports:
        for f in report["flags"]:
            key = (f.get("checklist_id"), f.get("file"), f.get("section"))
            if key in seen:
                continue
            seen.add(key)
            flags.append({**f, "path": f.get("path") if str(f.get("path", "")).startswith(path) else path})
    merged = {"case_id": case_id, "system": system, "flags": flags}
    from agent.report import validate
    validate(merged)
    return merged


class PlanReviewerV2(dspy.Module):
    """V2: the V1.1 checklist path and the value path, merged. forward(case_dir) ->
    dspy.Prediction(flags, report, verdicts, value_log)."""

    name = "v2"

    def __init__(self):
        super().__init__()
        self.checklist = PlanReviewer()
        self.values = ValueReviewer()

    def forward(self, case_dir: str) -> dspy.Prediction:
        a = self.checklist(case_dir=case_dir)
        b = self.values(case_dir=case_dir)
        report = merge_reports(a.report["case_id"], self.name, ("checklist", a.report), ("value", b.report))
        return dspy.Prediction(flags=report["flags"], report=report, verdicts=a.verdicts, value_log=b.value_log)
