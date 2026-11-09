"""The locator: the model supplies the words, code supplies the place.

A finding only scores if it points at the right place (file + section, or page +/-1), and models
guess sections and pages badly in long text. So a draft flag carries an exact quote, and anchor()
finds that quote in the plan and reads the place off the PlanIndex. It tries, in order:

  1 quote    the quote (or its longest '...'-separated fragment) is found
             -> file, nearest numbered section, every page the quote spans
  2 section  no quote match, but the model's file + section exist (the normal case for missing
             content, which has nothing to quote) -> that section and the page its heading is on
  3 model    the model's file + pages exist in that file -> kept as given
  4 none     nothing usable -> the file alone, or nothing (still enough for a deleted document)

The rung that worked is recorded as PlacedFlag.how, and the model's own guess is kept too.
A quote that can't be found anywhere is also a warning sign that it was made up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.plan import MIN_QUOTE_CHARS, Location, Match, PlanIndex, normalize
from agent.report import FlagDraft

_ELLIPSIS = re.compile(r"\.\.\.+|…|\[\.\.\.\]")


@dataclass(frozen=True)
class PlacedFlag:
    draft: FlagDraft
    location: Location | None
    how: str                      # "quote" | "section" | "model" | "none"

    @property
    def model_location(self) -> Location | None:
        d = self.draft
        if not (d.file or d.section or d.pages):
            return None
        return Location(d.file, d.section, tuple(d.pages))


def anchor(draft: FlagDraft, index: PlanIndex) -> PlacedFlag:
    """Place one draft flag (see the module docstring for the four rungs)."""
    if draft.quote:
        hits = index.find(draft.quote) or index.find(longest_fragment(draft.quote))
        if hits:
            m = pick(hits, index, prefer_file=draft.file, prefer_section=draft.section)
            return PlacedFlag(draft, index.locate(m.file, m.start, m.end), "quote")
    if draft.file and draft.section and draft.file in index.files:
        section = str(draft.section).lstrip("§ ").strip()
        if index.section_exists(draft.file, section):
            return PlacedFlag(draft, index.section_location(draft.file, section), "section")
    if draft.file and draft.pages and draft.file in index.files and index.pages_exist(draft.file, draft.pages):
        return PlacedFlag(draft, Location(draft.file, None, tuple(draft.pages)), "model")
    return PlacedFlag(draft, Location(draft.file) if draft.file else None, "none")


def pick(hits: list[Match], index: PlanIndex, prefer_file: str | None = None,
         prefer_section: str | None = None) -> Match:
    """One match out of several: the model's file first, then the section it named, then the first."""
    pool = [m for m in hits if m.file == prefer_file] or hits
    if prefer_section:
        want = str(prefer_section).lstrip("§ ").strip()
        same = [m for m in pool if (index.section_at(m.file, m.start) or "") == want]
        if same:
            return same[0]
    return pool[0]


def longest_fragment(quote: str) -> str:
    """The longest piece of a quote the model shortened with '...'."""
    parts = [p.strip() for p in _ELLIPSIS.split(quote)]
    parts = [p for p in parts if len(normalize(p)) >= MIN_QUOTE_CHARS]
    return max(parts, key=len) if parts else ""
