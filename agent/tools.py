"""The search functions as dspy.Tools, bound to one case.

    from agent.tools import make_tools
    tools = make_tools("bench/cases/thacker_pass_clean/TP-000")
    tools["search_plan"](query="riparian springs mitigation")     # try it in the notebook
    agent = dspy.ReAct(SomeSignature, tools=list(tools.values()))  # what V3 will do

Each tool returns plain text a model (or a person) can read: every passage starts with its place
(file · section · pages · heading) so it can be cited.
"""

from __future__ import annotations

from pathlib import Path

import dspy

from agent.plan import PlanIndex
from agent.search import Passage, PlanSearch


def render_passages(passages: list[Passage]) -> str:
    """Numbered passages with their snippets (short; for tools and people):
    "[1] body.md · §5.3 · p.100-101 · 5.3 Spill Contingency Plan\\n### 5.3 Spill Contingency Plan The …" """
    if not passages:
        return "No passages found."
    return "\n\n".join(f"[{i}] {p.header()}\n{p.snippet}" for i, p in enumerate(passages, 1))


def render_for_judge(passages: list[Passage], search: PlanSearch, max_chars: int = 2500) -> str:
    """Numbered passages with their FULL text (each cut at max_chars), for the V1 judge. A passage
    whose section has numbered sub-sections lists them, so an introduction ("Specific activities are
    described in the following sections") isn't mistaken for the whole treatment:
    "[1] body.md · §6.5 · p.111 · 6.5 Facility Reclamation Practices\\n<chunk text>\\n
     Sub-sections: 6.5.1 Decommissioning of Facilities (p.112); 6.5.2 Open Pit (p.112); …" """
    if not passages:
        return "No passages found."
    blocks = []
    for i, p in enumerate(passages, 1):
        block = f"[{i}] {p.header()}\n{search.passage_text(p, max_chars)}"
        subs = subsections(search.index, p.file, p.section)
        if subs:
            block += "\nSub-sections: " + "; ".join(subs)
        blocks.append(block)
    return "\n\n".join(blocks)


def subsections(index: PlanIndex, file: str, section: str | None, limit: int = 12) -> list[str]:
    """The direct numbered children of a section, with pages:
    subsections(ix, "body.md", "6.5") -> ["6.5.1 Decommissioning of Facilities (p.112)", …]"""
    if not section:
        return []
    depth = section.count(".") + 1
    out = [f"{h.title} (p.{index.page_at(file, h.offset)})" for h in index.outline(file)
           if h.number and h.number.startswith(section + ".") and h.number.count(".") == depth]
    return out[:limit] + ([f"… {len(out) - limit} more"] if len(out) > limit else [])


def plan_map(index: PlanIndex) -> str:
    """A compact map of the whole plan from its real headings: each file's title, the body's numbered
    sections down to x.y, and each appendix's top-level sections, with pages (~1,500 tokens). A deleted
    section simply isn't in it, so it leaks nothing.
    "body.md: Thacker Pass Project – Plan of Operations …\\n  1 Operator Information (p.19)\\n  1.1 …" """
    lines = []
    for f in index.files:
        heads = index.outline(f)
        lines.append(f"{f}: {next((h.title for h in heads if h.level == 1), f)}")
        for h in heads:
            if h.number and (h.number.count(".") <= 1 if f == "body.md" else h.number.count(".") == 0):
                lines.append(f"  {h.title} (p.{index.page_at(f, h.offset)})")
    return "\n".join(lines)


def render_missing_refs(refs: list) -> str:
    """'"Section 6.5.16" in body.md §6.5.6 p.114: target not found' per line, or a one-line all-clear."""
    if not refs:
        return "Every referenced section and appendix exists."
    return "\n".join(f'"{r.text}" in {r.location.file} §{r.location.section or "-"} '
                     f"p.{','.join(map(str, r.location.pages))}: target not found" for r in refs)


def make_tools(case: str | Path | PlanIndex) -> dict[str, dspy.Tool]:
    """The five plan tools for one case, by name."""
    index = case if isinstance(case, PlanIndex) else PlanIndex.load(case)
    search = PlanSearch(index)

    def search_plan(query: str, k: int = 5) -> str:
        """Search the plan for passages about a topic (BM25 keyword search). Returns up to k
        passages, best first, each with its file, section and pages."""
        return render_passages(search.search(query, k=min(k, 10)))

    def grep(phrase: str) -> str:
        """Find every place an exact phrase or number appears in the plan (case, spacing and
        thousands separators ignored). Use it to find all copies of a value or an exact name."""
        return render_passages(search.grep(phrase, k=20))

    def read_section(file: str, section: str) -> str:
        """Read one numbered section of one file, e.g. file='body.md', section='3.5'."""
        return search.read_section(file, section)

    def outline(file: str) -> str:
        """The headings of one file with their section numbers and first page: the plan's map."""
        lines = []
        for h in index.outline(file):
            page = index.page_at(file, h.offset)
            lines.append(f"{'  ' * max(h.level - 2, 0)}{h.title}  (p.{page})" if page else h.title)
        return "\n".join(lines) or f"{file} has no headings."

    def missing_references() -> str:
        """References in this plan to sections or appendices that do not exist in it."""
        return render_missing_refs(search.missing_refs())

    fns = [search_plan, grep, read_section, outline, missing_references]
    return {f.__name__: dspy.Tool(f) for f in fns}
