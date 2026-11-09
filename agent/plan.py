"""PlanIndex: a map of one plan (one case folder).

    index = PlanIndex.load("bench/cases/thacker_pass_clean/TP-021")
    index.files                         # ["body.md", "appendix_A.md", ...]  (body first)
    index.locate("body.md", start, end) # -> Location("body.md", "3.5", (51,))
    index.find("| West WRSF | 37.2 |")  # -> [Match("body.md", start, end)]  (original offsets)
    index.chunks                        # bite-size pieces for search, each with its place
    index.outline("body.md")            # the headings: the plan's map (it has no table of contents)

It reads the case's Markdown files once and builds four things per file:

  pages     where each <!-- page N | printed P --> marker sits     "what PDF page is this on?"
  headings  where each Markdown heading starts, and its number    "what section is this in?"
  chunks    one heading's span each, long ones split at page       "what should search rank?"
            breaks; every chunk carries file, section and pages
  flat      the text with markers removed, whitespace collapsed,   "where does this quote appear?"
            quotes/dashes normalized and lowercased, plus a map
            from each flat character back to its original offset

Section rule: a span's section is the number of the deepest *numbered* heading above it. An
unnumbered heading ("##### Tailings") takes its numbered parent's number ("3.5"); headings with
no numbered parent ("## Summary") give None. Where the nearest heading is numbered this matches
bench/build_cases.py: locate(), which tests/test_agent.py checks against every answer-key location.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = ROOT / "bench" / "cases"

# Same marker format as bench/build_cases.py (kept as a copy: agent code never imports the bench).
PAGE_MARKER = re.compile(r"<!-- page (\d+)(?: \| printed ([^ ]+))? -->")
HEADING = re.compile(r"^(#{1,6}) (.+)$", re.M)
SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\b")

MAX_CHUNK_CHARS = 4000   # a span longer than this is split at page breaks (a page is ~2,000 chars)
MIN_QUOTE_CHARS = 8      # shorter quotes match too many places to mean anything

# Characters normalized before matching quotes: curly quotes, dashes, odd spaces; '*' (bold) dropped.
_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ",
    "*": None,
})


# ------------------------------------------------------------------------------------- types

@dataclass(frozen=True)
class Location:
    """One spot in the plan, in the same shape the scorer uses: a file, maybe a section, maybe pages.

    Location("body.md", "3.5", (51,))   Location("appendix_E.md")   # whole file, no section/pages
    """
    file: str | None
    section: str | None = None
    pages: tuple[int, ...] = ()


@dataclass(frozen=True)
class Heading:
    """One Markdown heading. E.g. in TP-021 body.md:
    Heading("body.md", 89150, 3, "3.5 Waste Rock Storage Facilities", number="3.5", section="3.5", path="… > 3 Proposed Operations > 3.5 Waste Rock …")
    Heading("body.md", 109680, 5, "Tailings", number=None, section="3.5.1", path="… > 3.5.1 Mine Waste Rock … > Tailings")
    """
    file: str
    offset: int            # where the heading line starts
    level: int             # 1 for "#", 2 for "##", ...
    title: str             # "3.5 Waste Rock Storage Facilities"
    number: str | None     # its own number ("3.5"), or None if unnumbered
    section: str | None    # its number, or its numbered parent's number
    path: str              # "Thacker Pass … > 3 Description of Operations > 3.5 Waste Rock …"


@dataclass(frozen=True)
class Chunk:
    """A bite-size piece of the plan for search, labelled with its place. E.g.
    Chunk(id=68, file="body.md", start=89150, end=91844, section="3.5",
          heading="3.5 Waste Rock Storage Facilities", path="… > 3.5 Waste Rock …", pages=(51, 52))
    """
    id: int
    file: str
    start: int             # original offsets into the file text
    end: int
    section: str | None
    heading: str           # the heading this chunk sits under ("" before the first heading)
    path: str              # heading path, used as the chunk's "title" by search
    pages: tuple[int, ...]

    @property
    def location(self) -> Location:
        """chunk 68 -> Location("body.md", "3.5", (51, 52))"""
        return Location(self.file, self.section, self.pages)


@dataclass(frozen=True)
class Match:
    """Where a quote was found, as original offsets [start, end): Match("body.md", 90346, 90366)."""
    file: str
    start: int
    end: int


@dataclass
class _File:
    """Everything built for one file. E.g. body.md: pages=[(328, 5, None), (917, 6, "i"), …,
    (88678, 51, "33"), …]; headings=[Heading…]; flat="thacker pass project …"; back=[…]."""
    name: str
    text: str
    pages: list[tuple[int, int, str | None]] = field(default_factory=list)  # (offset, pdf page, printed)
    headings: list[Heading] = field(default_factory=list)
    flat: str = ""
    back: list[int] = field(default_factory=list)                           # flat index -> original offset


# --------------------------------------------------------------------------------- the index

class PlanIndex:
    """A map of one case: pages, headings, chunks and a flat copy of every file."""

    def __init__(self, files: dict[str, str], name: str = ""):
        """Build from {file name: text}, e.g. PlanIndex({"body.md": "...", "appendix_A.md": "..."}, "TP-021").
        Usually called through load(). Result: 12 files, 515 chunks for a Thacker Pass case."""
        self.name = name
        order = sorted(files, key=lambda f: (f != "body.md", f))   # body first, then appendices, figures
        self._files = {f: _build_file(f, files[f]) for f in order}
        self.chunks: list[Chunk] = []
        for f in self._files.values():
            self.chunks += _chunks(f, first_id=len(self.chunks))

    @classmethod
    def load(cls, case_dir: str | Path) -> PlanIndex:
        """Read one case folder. Only folders under bench/cases/ are allowed, and only the files
        the plan's cases.json lists for that case (or every .md file if there is no listing).

        load("bench/cases/thacker_pass_clean/TP-021")  -> PlanIndex (name "TP-021", 12 files)
        load("bench/cases/thacker_pass_clean/TP-001")  -> 11 files (appendix_E.md was deleted)
        load("Data/Thacker Pass")                       -> ValueError (outside bench/cases/)
        """
        path = Path(case_dir)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        if CASES_ROOT.resolve() not in path.parents:
            raise ValueError(f"PlanIndex only reads case folders under {CASES_ROOT}, not {path}")
        if not path.is_dir():
            raise FileNotFoundError(path)
        names = _listed_files(path) or sorted(p.name for p in path.glob("*.md"))
        return cls({n: (path / n).read_text(encoding="utf-8") for n in names}, name=path.name)

    # --- basic access -----------------------------------------------------------------------

    @property
    def files(self) -> list[str]:
        """["body.md", "appendix_A.md", "appendix_B_summary.md", …, "figures.md"]  (body first)"""
        return list(self._files)

    def text(self, file: str) -> str:
        """The raw file text, markers included: text("body.md")[90346:90366] -> "| West WRSF | 37.2 |" """
        return self._file(file).text

    def chunk_text(self, chunk: Chunk) -> str:
        """The raw text of a chunk: chunk 68 -> "### 3.5 Waste Rock Storage Facilities\\n\\nUp to 45.9 M CY …" """
        return self._file(chunk.file).text[chunk.start:chunk.end]

    def outline(self, file: str) -> list[Heading]:
        """All headings of a file, in order: outline("body.md")[4].title -> "1 Operator Information" """
        return list(self._file(file).headings)

    # --- where is this? ---------------------------------------------------------------------

    def page_at(self, file: str, offset: int) -> int | None:
        """The PDF page of the last page marker at or before offset (None before the first marker).
        body.md markers: page 51 at 88678, page 52 at 91278  ->  page_at("body.md", 90346) -> 51"""
        page = None
        for off, pdf, _ in self._file(file).pages:
            if off > offset:
                break
            page = pdf
        return page

    def section_at(self, file: str, offset: int) -> str | None:
        """The section of the last heading at or before offset (its number or its numbered parent's).
        section_at("body.md", 90346) -> "3.5"  (heading "3.5 Waste Rock…" starts at 89150)
        section_at("body.md", 109700) -> "3.5.1" ("##### Tailings" takes its parent's number)"""
        section = None
        for h in self._file(file).headings:
            if h.offset > offset:
                break
            section = h.section
        return section

    def locate(self, file: str, start: int, end: int | None = None) -> Location:
        """File, ONE section and EVERY page that the span [start, end) touches.

        pages:   all pages the span touches (a span crossing a page marker gets both pages).
        section: the section holding MOST of the span's characters. A Location (and a report flag)
                 has one section, and the scorer compares sections before pages, so a span that
                 starts with the last words of one section and runs into the next is labelled by
                 the section it is mostly in. Ties go to the section where the span starts.

        locate("body.md", 90346, 90366) -> Location("body.md", "3.5", (51,))
        locate("body.md", 90346, 91300) -> Location("body.md", "3.5", (51, 52))  # crosses the page-52 marker
        locate("body.md", 89100, 89400) -> Location("body.md", "3.5", (51,))     # 50 chars in 3.4.3, 250 in 3.5
        locate("body.md", 89000, 89200) -> Location("body.md", "3.4.3", (51,))   # 150 in 3.4.3, 50 in 3.5
        locate("body.md", 89050, 89250) -> Location("body.md", "3.4.3", (51,))   # 100 / 100 tie -> where it starts
        (The "### 3.5 Waste Rock Storage Facilities" heading starts at 89150.)
        """
        end = start + 1 if end is None or end <= start else end
        first = self.page_at(file, start)
        pages = [first] if first is not None else []
        pages += [pdf for off, pdf, _ in self._file(file).pages if start < off < end and pdf not in pages]
        return Location(file, self._main_section(file, start, end), tuple(pages))

    def _main_section(self, file: str, start: int, end: int) -> str | None:
        """The section with the most characters of [start, end); ties go to the earliest.
        Headings inside the span cut it into pieces; each piece counts toward its own section.
        _main_section("body.md", 89100, 89400) -> "3.5"    pieces: 89100–89150 (3.4.3, 50), 89150–89400 (3.5, 250)"""
        cuts = [h.offset for h in self._file(file).headings if start < h.offset < end]
        edges = [start] + cuts + [end]
        sizes: dict[str | None, int] = {}
        for a, b in zip(edges, edges[1:]):
            s = self.section_at(file, a)
            sizes[s] = sizes.get(s, 0) + (b - a)
        return max(sizes, key=sizes.get)   # dicts keep insertion order, so a tie keeps the earliest

    def section_exists(self, file: str, number: str) -> bool:
        """section_exists("body.md", "3.5") -> True;  section_exists("body.md", "99.9") -> False"""
        return file in self._files and any(h.number == number for h in self._files[file].headings)

    def section_location(self, file: str, number: str) -> Location | None:
        """The place of a numbered section: the section and the page its heading is on.
        section_location("body.md", "3.5") -> Location("body.md", "3.5", (51,));  unknown number -> None"""
        for h in self._file(file).headings:
            if h.number == number:
                page = self.page_at(file, h.offset)
                return Location(file, number, (page,) if page is not None else ())
        return None

    def pages_exist(self, file: str, pages: list[int] | tuple[int, ...]) -> bool:
        """pages_exist("body.md", [51, 52]) -> True;  [9999] -> False;  [] -> False"""
        have = {pdf for _, pdf, _ in self._file(file).pages} if file in self._files else set()
        return bool(pages) and all(p in have for p in pages)

    # --- find a quote -----------------------------------------------------------------------

    def find(self, quote: str, files: list[str] | None = None) -> list[Match]:
        """Every place the quote appears, ignoring case, whitespace, page markers, curly quotes,
        dash styles and bold markers. Returns original offsets, so locate() can place each match.
        find("| West WRSF | 37.2 |")  -> [Match("body.md", 90346, 90366)]
        find("| WEST wrsf |   37.2 |") -> the same (case and spacing don't matter)
        find("the")                     -> []  (shorter than MIN_QUOTE_CHARS)"""
        q = normalize(quote)
        if len(q) < MIN_QUOTE_CHARS:
            return []
        out = []
        for name in files or self.files:
            if name not in self._files:
                continue
            f = self._files[name]
            i = f.flat.find(q)
            while i != -1:
                out.append(Match(name, f.back[i], f.back[i + len(q) - 1] + 1))
                i = f.flat.find(q, i + 1)
        return out

    def find_pattern(self, pattern: re.Pattern, files: list[str] | None = None) -> list[Match]:
        """Every match of a regex over the flat (normalized) text, as original offsets. The
        pattern must be written for normalized text: lowercase, single spaces, straight quotes.
        find_pattern(re.compile(r"west wrsf \\| 37\\.2")) -> [Match("body.md", 90348, 90364)]"""
        out = []
        for name in files or self.files:
            if name not in self._files:
                continue
            f = self._files[name]
            for m in pattern.finditer(f.flat):
                if m.end() > m.start():
                    out.append(Match(name, f.back[m.start()], f.back[m.end() - 1] + 1))
        return out

    def chunk_at(self, file: str, offset: int) -> Chunk | None:
        """The chunk containing offset (or the nearest one before it, for offsets in skipped
        whitespace or markers). chunk_at("body.md", 90346) -> chunk 68 (89150–91844, §3.5, pages 51–52)"""
        best = None
        for c in self._chunks_by_file().get(file, []):
            if c.start > offset:
                break
            best = c
        return best

    # --- internals --------------------------------------------------------------------------

    def _chunks_by_file(self) -> dict[str, list[Chunk]]:
        """{"body.md": [chunk 0, chunk 1, …], "appendix_A.md": […], …}, built on first use."""
        if not hasattr(self, "_by_file"):
            self._by_file: dict[str, list[Chunk]] = {}
            for c in self.chunks:
                self._by_file.setdefault(c.file, []).append(c)
        return self._by_file

    def _file(self, file: str) -> _File:
        """_file("body.md") -> its _File;  _file("nope.md") -> KeyError listing the case's files"""
        try:
            return self._files[file]
        except KeyError:
            raise KeyError(f"{file!r} is not in this case. Files: {', '.join(self.files)}") from None


# --------------------------------------------------------------------------------- building

def normalize(text: str) -> str:
    """The text as quotes are compared: markers dropped, characters normalized, whitespace
    collapsed to single spaces, lowercase. Used for both the plan (flat copy) and quotes.
    normalize('The  “West”\\nWRSF <!-- page 51 | printed 33 --> is **big**') -> 'the "west" wrsf is big'"""
    text = PAGE_MARKER.sub(" ", text).translate(_CHAR_MAP).lower()
    return " ".join(text.split())


# Same as _CHAR_MAP but length-preserving, so offsets survive: '*' becomes a NUL to skip later.
_CHAR_MAP_SAME_LEN = {k: (v if v is not None else "\x00") for k, v in _CHAR_MAP.items()}
_RUN = re.compile(r"\S+")


def _flatten(text: str) -> tuple[str, list[int]]:
    """normalize(text), plus back[i] = the original offset of flat character i.

    Page markers are blanked with spaces of the same length and characters are normalized one for
    one, so every non-space run still sits at its original offsets; runs are then joined with
    single spaces. (A space maps back to the end of the run before it.)

    _flatten("Ab  <!-- page 7 -->\\nCd") -> ("ab cd", [0, 1, 1, 20, 21])
             a→0  b→1  " "→1  c→20  d→21   (the marker and newline are skipped)"""
    blanked = PAGE_MARKER.sub(lambda m: " " * len(m.group(0)), text)
    same = blanked.translate(_CHAR_MAP_SAME_LEN)
    low = same.lower()
    if len(low) != len(same):                 # a rare character whose lowercase is longer
        low = "".join(c.lower() if len(c.lower()) == 1 else c for c in same)
    parts: list[str] = []
    back: list[int] = []
    for m in _RUN.finditer(low):
        run, a = m.group(0), m.start()
        if "\x00" in run:                     # drop bold markers inside this run
            kept = [(c, a + i) for i, c in enumerate(run) if c != "\x00"]
            if not kept:
                continue
            run = "".join(c for c, _ in kept)
            offsets = [o for _, o in kept]
        else:
            offsets = range(a, a + len(run))
        if parts:
            parts.append(" ")
            back.append(back[-1])
        parts.append(run)
        back.extend(offsets)
    return "".join(parts), back


def _build_file(name: str, text: str) -> _File:
    """Build one file's tables: pages, headings (with section + path, via a stack of open
    headings) and the flat copy. E.g. "### 3.5 Waste…" then "#### 3.5.1 Mine…" then "##### Tailings"
    -> the Tailings heading gets section "3.5.1" and path "… > 3.5 … > 3.5.1 … > Tailings"."""
    f = _File(name, text)
    f.pages = [(m.start(), int(m.group(1)), m.group(2)) for m in PAGE_MARKER.finditer(text)]
    stack: list[Heading] = []   # the open headings above the current point, shallowest first
    for m in HEADING.finditer(text):
        level, title = len(m.group(1)), m.group(2).strip()
        num = SECTION_NUMBER.match(title)
        number = num.group(1) if num else None
        stack = [h for h in stack if h.level < level]
        parent_section = next((h.section for h in reversed(stack) if h.section), None)
        path = " > ".join([h.title for h in stack] + [title])
        h = Heading(name, m.start(), level, title, number, number or parent_section, path)
        stack.append(h)
        f.headings.append(h)
    f.flat, f.back = _flatten(text)
    return f


def _chunks(f: _File, first_id: int) -> list[Chunk]:
    """One chunk per heading span (plus the text before the first heading); a span longer than
    MAX_CHUNK_CHARS is split at page markers, grouping pages up to that size. Whitespace-only
    pieces are dropped.
    body.md heading "3.5 Waste Rock…" (89150) to the next heading (91844) -> one chunk (2,694 chars)."""
    bounds = [(0, None)] + [(h.offset, h) for h in f.headings]
    if len(bounds) > 1 and bounds[1][0] == 0:
        bounds = bounds[1:]                   # the file starts with a heading
    out: list[Chunk] = []
    for i, (start, h) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(f.text)
        for a, b in _split_at_pages(f, start, end):
            body = PAGE_MARKER.sub("", f.text[a:b])
            if not body.strip():
                continue
            pages = _pages_in(f, a, b)
            out.append(Chunk(first_id + len(out), f.name, a, b,
                             h.section if h else None, h.title if h else "", h.path if h else "", pages))
    return out


def _split_at_pages(f: _File, start: int, end: int) -> list[tuple[int, int]]:
    """Cut a long span at page markers into pieces of at most MAX_CHUNK_CHARS (whole pages each).
    Span 0–9000, markers at 2000, 4500, 7000 -> [(0, 2000), (2000, 4500), (4500, 7000), (7000, 9000)]
    Span 0–5000, markers at 1500, 3000       -> [(0, 3000), (3000, 5000)]  (two pages fit in one piece)
    A span of 4000 chars or less -> [(start, end)] unchanged."""
    if end - start <= MAX_CHUNK_CHARS:
        return [(start, end)]
    cuts = [off for off, _, _ in f.pages if start < off < end]
    edges = [start] + cuts + [end]
    pieces, a = [], start
    for prev, nxt in zip(edges, edges[1:]):
        if nxt - a > MAX_CHUNK_CHARS and prev > a:
            pieces.append((a, prev))
            a = prev
    pieces.append((a, end))
    return pieces


def _pages_in(f: _File, a: int, b: int) -> tuple[int, ...]:
    """Every page the span [a, b) touches: _pages_in(body.md, 89150, 91844) -> (51, 52)."""
    page = None
    for off, pdf, _ in f.pages:
        if off > a:
            break
        page = pdf
    pages = [page] if page is not None else []
    for off, pdf, _ in f.pages:
        if a < off < b and pdf not in pages:
            pages.append(pdf)
    return tuple(pages)


def _listed_files(case_dir: Path) -> list[str] | None:
    """The files cases.json lists for this case, or None if there is no listing.
    TP-001 -> ["appendix_A.md", …, "appendix_D.md", "appendix_F.md", …, "figures.md"]  (no appendix_E.md)"""
    listing = case_dir.parent / "cases.json"
    if not listing.exists():
        return None
    for c in json.loads(listing.read_text(encoding="utf-8")).get("cases", []):
        if c.get("case_id") == case_dir.name:
            return list(c.get("files", []))
    return None
