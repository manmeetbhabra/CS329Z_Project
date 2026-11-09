"""V1: go through the checklist. For each of the 51 requirements, search the plan and let an AI
judge decide whether the plan meets it, and list each specific problem with an exact quote.

    from agent.config import configure
    from agent.systems.v1 import PlanReviewer
    configure("dev")
    pred = PlanReviewer()(case_dir="bench/cases/thacker_pass_clean/TP-012")
    pred.flags      # report flags, placed by the locator (one per finding)
    pred.verdicts   # one row per item, including "present": why it decided what it did

Per case: build the plan map and search every item (no AI, fast), then 51 judge calls in parallel,
then each finding becomes a FlagDraft that the locator places and to_report() writes.

V1.1 (2026-11-08), after reviewing V1's flags on the clean control (Agent_Plan_V1.md §7-8):
  * the judge returns a LIST of findings per item (one verdict per item could flag only one place,
    so a planted flaw lost out to a standing weakness under the same item);
  * each passage lists its sub-sections, so an introduction isn't mistaken for the whole treatment
    (a whole-plan map is available with use_map=True, but didn't help: off by default);
  * the instructions include short "not a problem" examples for calibration.
"""

from __future__ import annotations

from typing import Literal

import dspy
from pydantic import BaseModel

from agent.anchor import anchor
from agent.checklist import Item, load_checklist
from agent.plan import PlanIndex
from agent.queries import load_item_queries
from agent.report import FlagDraft, to_report
from agent.search import Passage, PlanSearch
from agent.tools import plan_map, render_for_judge, render_missing_refs

Verdict = Literal["present", "problem", "not_applicable"]
Problem = Literal["vague", "missing", "inconsistent", "implausible"]
PROBLEMS = ("vague", "missing", "inconsistent", "implausible")     # finding types that raise a flag
MAX_FINDINGS = 3                                                    # per item, most important first


class Finding(BaseModel):
    """One specific problem, tied to one quoted spot in the plan."""
    status: Problem
    quote: str                  # exact words from one passage
    file: str                   # e.g. "body.md"
    section: str                # e.g. "6.4.9", or "" if none
    reason: str
    confidence: float           # 0 to 1


class JudgeItem(dspy.Signature):
    """You are a BLM reviewer checking a draft Plan of Operations for a mine on federal land
    (43 CFR 3809) against ONE requirement, using the numbered passages shown and the plan map.

    Give an overall verdict:
      present        - the plan meets the requirement, as a BLM reviewer would accept it;
      problem        - at least one specific deficiency (list it under findings);
      not_applicable - the requirement's 'Applies when' condition is false for this project.
    For 'problem', list each distinct deficiency as a finding (most important first, at most 3),
    each tied to ONE spot in the plan:
      vague        - hedged wording stands in for the specifics the requirement asks for ('as
                     appropriate', 'as necessary', 'to be determined', 'may ... where practicable',
                     'will be determined at a later date'), failing its 'Too vague if' test;
      missing      - the plan does not provide it: the topic is named but never described, or a
                     missing reference shows the section/appendix holding it was cited but is absent;
      inconsistent - two passages give conflicting values or facts for the same thing;
      implausible  - a stated value is one no mining engineer or reviewer would accept.

    How BLM plans are written: a plan summarizes, and routinely relies on companion documents - its
    own appendices, the state Water Pollution Control Permit, design reports, a reclamation cost
    estimate submitted separately. A firm commitment naming the document that holds the details is
    acceptable. The passages are only parts of a long plan: use the plan map and each passage's
    'Sub-sections' line - if a section that would hold the details exists (e.g. an introduction says
    specifics follow in sub-sections, or a list names a facility that has its own section), do not
    call it vague or missing just because those details are not among the passages.

    These are NOT problems (examples): a stated value next to one soft word ("quarterly samples will
    typically be collected"); an engineering contingency ("if needed, active evaporation will be
    used"); site staff listed as "to be determined" in a pre-operations emergency or spill plan while
    the plan names its contact person elsewhere; a deferral to a named permit or plan that will be
    submitted to the agencies.

    These ARE problems even when the topic is otherwise covered: a missing reference to a section
    or appendix that would hold what this requirement asks for (report 'missing', quoting the
    sentence that cites it); a specific number - a design storm, capacity, slope, height, rate, count
    or date - that conflicts with another passage or that no engineer would accept (quote it).
    Report a problem only when you can quote the wording that shows it.

    Quote the plan exactly: copy a short phrase or sentence word for word from one passage - the
    hedged wording, each of the conflicting or implausible values, or for 'missing' the passage that
    names or cites the absent content. Give the file and section number of that passage."""

    requirement: str = dspy.InputField(desc="the requirement: id, citation, text, applies when, too vague if")
    plan_map: str = dspy.InputField(desc="the plan's files and numbered sections with pages")
    passages: str = dspy.InputField(desc="numbered passages, each headed 'file · §section · pages · heading'")
    missing_references: str = dspy.InputField(desc="references in this plan to sections or appendices that do not exist")
    verdict: Verdict = dspy.OutputField()
    findings: list[Finding] = dspy.OutputField(desc="the deficiencies; empty unless verdict is 'problem'")
    reason: str = dspy.OutputField(desc="one or two sentences on the verdict")
    confidence: float = dspy.OutputField(desc="0 to 1: how sure you are of the verdict")


class PlanReviewer(dspy.Module):
    """The V1 agent. forward(case_dir) -> dspy.Prediction(flags, report, verdicts)."""

    name = "v1"

    def __init__(self, k: int = 10, min_confidence: float = 0.5, threads: int = 8, passage_chars: int = 2500,
                 use_map: bool = False):
        super().__init__()
        # The plan map (agent.tools.plan_map) is off by default: on the full bench it doubled the cost
        # and caught 9 genuine flaws vs 11 without it (Agent_Plan_V1.md §8). Kept as an option.
        self.use_map = use_map
        self.judge = dspy.ChainOfThought(JudgeItem)     # the only part that calls an LLM
        self.items: tuple[Item, ...] = load_checklist()
        self.queries = load_item_queries()
        self.k, self.min_confidence, self.threads, self.passage_chars = k, min_confidence, threads, passage_chars

    def forward(self, case_dir: str) -> dspy.Prediction:
        index = PlanIndex.load(case_dir)
        search = PlanSearch(index)
        refs = render_missing_refs(search.missing_refs())
        the_map = plan_map(index) if self.use_map else "(not provided)"
        found = {i.id: search.hybrid(self.queries[i.id], k=self.k) for i in self.items}       # no AI
        inputs = [dspy.Example(requirement=i.render(), plan_map=the_map, missing_references=refs,
                               passages=render_for_judge(found[i.id], search, self.passage_chars)
                               ).with_inputs("requirement", "plan_map", "passages", "missing_references")
                  for i in self.items]
        verdicts = self.judge.batch(inputs, num_threads=self.threads, disable_progress_bar=True,
                                    max_errors=len(inputs))                                      # 51 judge calls
        placed, log = [], []
        for item, v in zip(self.items, verdicts):
            row = verdict_row(item, v, found[item.id])
            log.append(row)
            for f in row["findings"]:
                if f["confidence"] < self.min_confidence:
                    continue
                draft = FlagDraft(checklist_id=item.id, type=f["status"], quote=f["quote"] or None,
                                  file=f["file"] or None, section=f["section"] or None,
                                  explanation=f["reason"], confidence=f["confidence"])
                p = anchor(draft, index)
                f["placed"] = {"how": p.how, "file": p.location.file if p.location else None,
                               "section": p.location.section if p.location else None,
                               "pages": list(p.location.pages) if p.location else []}
                placed.append(p)
        report = to_report(index.name, self.name, placed)
        return dspy.Prediction(flags=report["flags"], report=report, verdicts=log)


def _conf(x) -> float:
    try:
        return max(0.0, min(1.0, float(x or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def verdict_row(item: Item, v, passages: list[Passage]) -> dict:
    """One judge verdict as a plain dict for logs and the miss report. A failed call (v is None)
    becomes status "error" and never raises a flag. "status" is the verdict ("problem" becomes the
    first finding's type, so logs read like V1.0's); "findings" holds every problem found.
    -> {"item": "rec-riparian", "status": "missing", "confidence": 0.8, "reason": "...",
        "findings": [{"status", "quote", "file", "section", "reason", "confidence"}, ...],
        "passages": [{"file", "section", "pages"}, ...]}"""
    places = [{"file": p.file, "section": p.section, "pages": list(p.pages)} for p in passages]
    if v is None:
        return {"item": item.id, "status": "error", "confidence": 0.0, "reason": "judge call failed",
                "findings": [], "quote": "", "file": "", "section": "", "passages": places}
    findings = []
    for f in (getattr(v, "findings", None) or [])[:MAX_FINDINGS]:
        d = f.model_dump() if hasattr(f, "model_dump") else dict(f)
        if d.get("status") in PROBLEMS:
            findings.append({"status": d["status"], "quote": (d.get("quote") or "").strip(),
                             "file": (d.get("file") or "").strip(),
                             "section": str(d.get("section") or "").strip().lstrip("§ "),
                             "reason": (d.get("reason") or "").strip(), "confidence": _conf(d.get("confidence"))})
    verdict = str(getattr(v, "verdict", "error"))
    status = findings[0]["status"] if (verdict == "problem" and findings) else ("present" if verdict == "problem" else verdict)
    first = findings[0] if findings else {}
    return {"item": item.id, "status": status, "confidence": _conf(getattr(v, "confidence", 0.0)),
            "reason": (getattr(v, "reason", "") or "").strip(), "findings": findings if verdict == "problem" else [],
            "quote": first.get("quote", ""), "file": first.get("file", ""), "section": first.get("section", ""),
            "passages": places}
