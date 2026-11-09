"""PlanSearch: three ways to find text in a plan, because they find different things.

    search = PlanSearch(PlanIndex.load(case_dir))
    search.search("riparian springs mitigation", k=5)   # BM25: passages ABOUT a topic
    search.grep("32.7 M CY")                             # exact: EVERY place a phrase/number occurs
    search.refs()                                        # every "Section X" / "Appendix Y" mention
    search.missing_refs()                                # mentions of parts that don't exist here
    search.hybrid(["riparian mitigation", '"seeps"'])    # several queries merged into one top-k list

Every result is a Passage that carries its place (file, section, pages), so whoever reads it can
cite it, and the scorer can check it.

Why three methods:
  BM25 (agent/bm25.py, from HW1) ranks chunks by shared words, rare words counting more. It
    splits "32.7 M CY" into "32", "7", "m", "cy" and ignores order, so it cannot find an exact
    number or phrase.
  grep is Ctrl-F over the normalized text: case, spacing, quote and dash styles don't matter,
    and "5,252" matches "5252". It finds every copy of a value (what V2's consistency check
    needs) and exact names such as "Appendix E" or "NAC 534".
  Cross-references: plans point elsewhere ("see Section 6.5.16", "Appendix E"). A pointer to a
    section or appendix that doesn't exist in this case is a strong sign that something was
    deleted, and costs no AI to find.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.bm25 import BM25Index
from agent.plan import PAGE_MARKER, Chunk, Location, PlanIndex, normalize

SNIPPET_CHARS = 300

# "Section 6.5.16", "Sections 3.2 and 3.4" (first one), "§ 4.12". A first number above
# MAX_TOP_SECTION is a CFR/NAC/statute paragraph ("Section 3809.401", "Section 404"), not a plan section.
SECTION_REF = re.compile(r"\b(?:Sections?|§)\s*(\d+(?:\.\d+)*)", re.I)
APPENDIX_REF = re.compile(r"\bAppendix\s+([A-Z])\b")
MAX_TOP_SECTION = 20


@dataclass(frozen=True)
class Passage:
    """One search result: a chunk, its place, and a short snippet. E.g. (TP-000):
    Passage(chunk_id=143, file="body.md", section="5.3", heading="5.3 Spill Contingency Plan",
            pages=(100, 101), score=9.22, snippet="### 5.3 Spill Contingency Plan The Spill Contingency Plan (Appendix E) was devel…")
    """
    chunk_id: int
    file: str
    section: str | None
    heading: str
    pages: tuple[int, ...]
    score: float
    snippet: str

    @property
    def location(self) -> Location:
        """passage 143 -> Location("body.md", "5.3", (100, 101))"""
        return Location(self.file, self.section, self.pages)

    def header(self) -> str:
        """The place as one readable line:
        passage 143 -> "body.md · §5.3 · p.100-101 · 5.3 Spill Contingency Plan"
        passage 324 -> "appendix_E.md · no section · p.1 · Appendix E – Spill Contingency Plan" """
        where = [self.file, f"§{self.section}" if self.section else "no section"]
        if self.pages:
            where.append(f"p.{self.pages[0]}" + (f"-{self.pages[-1]}" if len(self.pages) > 1 else ""))
        if self.heading:
            where.append(self.heading)
        return " · ".join(where)


@dataclass(frozen=True)
class Ref:
    """One cross-reference: where it is, what it points to, and whether that exists in this case.
    Ref("appendix", "appendix_A.md", "Appendix A", Location("body.md", "1.6", (20,)), exists=True)
    Ref("section", "6.5.16", "Section 6.5.16", Location("body.md", "6.5.6", (114,)), exists=False)  # TP-002
    """
    kind: str             # "section" | "appendix"
    target: str           # "6.5.16" | "appendix_E.md"
    text: str             # the matched words, e.g. "Section 6.5.16"
    location: Location    # where the reference itself is
    exists: bool


class PlanSearch:
    """BM25 + grep + cross-references over one PlanIndex."""

    def __init__(self, index: PlanIndex):
        """PlanSearch(PlanIndex.load("bench/cases/thacker_pass_clean/TP-000")) -> BM25 over its 515 chunks
        (each indexed as "heading path + text"). Takes about 0.2 s."""
        self.index = index
        self.chunks = index.chunks
        # As in HW1's DocIndex: each chunk is indexed with its heading path as a title, so a query
        # naming the topic ("Spill Contingency Plan") also matches the sections under it.
        self.bm25 = BM25Index([c.path + "\n" + PAGE_MARKER.sub(" ", index.chunk_text(c)) for c in self.chunks])
        self._refs: list[Ref] | None = None

    # --- BM25 -----------------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[Passage]:
        """The k chunks that best match the query's words, best first (score = BM25).
        search("spill contingency plan", k=2) ->
            [Passage(324, "appendix_E.md", None, pages=(1,),       score=9.64, …),
             Passage(143, "body.md",       "5.3", pages=(100, 101), score=9.22, …)]"""
        return [self._passage(self.chunks[i], score, query) for i, score in self.bm25.search(query, k)]

    # --- exact match ----------------------------------------------------------------------

    def grep(self, phrase: str, k: int | None = None) -> list[Passage]:
        """Every chunk containing the exact phrase (normalized), in document order. Numbers match
        with or without thousands separators, and never inside a longer number ("32.7" won't match
        "132.7"). Score is always 1.0; the snippet is centred on the match.
        grep("5,252")     -> [Passage(68, "body.md", "3.5", (51, 52), …)]   (grep("5252") gives the same)
        grep("32.7 M CY") -> [body.md §3.5 p.51-52, appendix_C.md §3.2 p.13-14]   (every copy of the value)
        grep("x")         -> []   (too short)"""
        pattern = phrase_pattern(phrase)
        if pattern is None:
            return []
        seen, out = set(), []
        for m in self.index.find_pattern(pattern):
            c = self.index.chunk_at(m.file, m.start)
            if c is None or c.id in seen:
                continue
            seen.add(c.id)
            out.append(self._passage(c, 1.0, phrase, at=m.start))
        return out[:k] if k else out

    # --- cross-references -----------------------------------------------------------------

    def refs(self) -> list[Ref]:
        """Every 'Section X' and 'Appendix Y' mention in the plan, with whether its target exists.

        A section reference counts as existing if that section number is a heading in the same
        file or in body.md (appendices often cite the plan body). An appendix reference exists if
        the case has that appendix's file. "Section 3.0" is read as "3"; "Section 3809.401" (a CFR
        paragraph) is skipped. Computed once, then cached.
        refs() on TP-000 -> 124 Refs, starting  Ref("section", "6", "Section 6", Location("body.md", None, (8,)), True)"""
        if self._refs is None:
            appendices = {_appendix_letter(f): f for f in self.index.files if _appendix_letter(f)}
            out = []
            for file in self.index.files:
                text = self.index.text(file)
                for m in SECTION_REF.finditer(text):
                    number = re.sub(r"(\.0)+$", "", m.group(1).rstrip("."))   # "3.0" is how plans write "3"
                    if int(number.split(".")[0]) > MAX_TOP_SECTION:
                        continue
                    exists = self.index.section_exists(file, number) or self.index.section_exists("body.md", number)
                    out.append(Ref("section", number, m.group(0), self.index.locate(file, m.start(), m.end()), exists))
                for m in APPENDIX_REF.finditer(text):
                    letter = m.group(1)
                    if _appendix_letter(file) == letter:
                        continue                                   # an appendix naming itself
                    target = appendices.get(letter, f"appendix_{letter}.md")
                    out.append(Ref("appendix", target, m.group(0),
                                   self.index.locate(file, m.start(), m.end()), letter in appendices))
            self._refs = out
        return self._refs

    def missing_refs(self) -> list[Ref]:
        """References to sections or appendices that are not in this case.
        TP-000 -> [Ref("section", "1.5.4", …, Location("appendix_C.md", "1.5", (11,)), False)]  (a real gap in the plan)
        TP-001 -> adds two Refs to "appendix_E.md" (cited in body §4.12 and §5.3; the appendix was deleted)
        TP-002 -> adds three Refs to "6.5.16" (cited in body §6.5.6, §6.5.8, §6.5.13; the section was deleted)"""
        return [r for r in self.refs() if not r.exists]

    # --- several queries at once ----------------------------------------------------------

    def hybrid(self, queries: list[str], k: int = 6) -> list[Passage]:
        """Run each query (a "quoted" query is a grep, anything else BM25) and merge the results
        by reciprocal rank: a chunk found high up by several queries rises to the top.
        Each query adds 1/(rank + 60) to every chunk it finds (rank 0 = its best); score = the sum.
        hybrid(["riparian mitigation springs", '"seasonal springs"'], k=3) ->
            [Passage(167, body.md §6.4.9 "Riparian Areas and Springs",  score=0.0328),   # found by both
             Passage(23,  body.md §2.2.8 "Location of Surface Water…", score=0.03116),
             Passage(35,  body.md §2.3.4 "Seeps and Springs",          score=0.01639)]  # found by one"""
        scores: dict[int, float] = {}
        best: dict[int, Passage] = {}
        for q in queries:
            q = q.strip()
            if not q:
                continue
            exact = len(q) > 2 and q[0] == q[-1] == '"'
            results = self.grep(q[1:-1], k=max(k, 10)) if exact else self.search(q, k=max(k, 10))
            for rank, p in enumerate(results):
                scores[p.chunk_id] = scores.get(p.chunk_id, 0.0) + 1.0 / (rank + 60)
                best.setdefault(p.chunk_id, p)
        top = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
        return [Passage(**{**best[cid].__dict__, "score": round(scores[cid], 5)}) for cid in top]

    # --- reading --------------------------------------------------------------------------

    def read_section(self, file: str, number: str, max_chars: int = 6000) -> str:
        """The text of one numbered section (up to the next heading of the same or a higher level),
        page markers removed, cut at max_chars.
        read_section("body.md", "1.5") -> "### 1.5 Taxpayer Identification Number\\n\\nLNC will provide the Federal Tax …"
        read_section("body.md", "99")  -> KeyError"""
        heads = self.index.outline(file)
        for i, h in enumerate(heads):
            if h.number == number:
                end = next((n.offset for n in heads[i + 1:] if n.level <= h.level), len(self.index.text(file)))
                body = PAGE_MARKER.sub("", self.index.text(file)[h.offset:end]).strip()
                return body if len(body) <= max_chars else body[:max_chars] + "\n[... section continues ...]"
        raise KeyError(f"no section {number} in {file}")

    def passage_text(self, p: Passage, max_chars: int = 2500) -> str:
        """The passage's full chunk text for a reader (the V1 judge): page markers removed, blank-line
        runs squeezed, cut at max_chars with a note. Table rows and line breaks are kept.
        passage_text(passage 68) -> "### 3.5 Waste Rock Storage Facilities\\n\\nUp to 45.9 M CY of waste rock …" """
        chunk = self.chunks[p.chunk_id]
        raw = PAGE_MARKER.sub("", self.index.chunk_text(chunk))
        # A chunk that is little more than its heading ("#### 2.3.6 Wildlife") keeps its content in
        # the sub-sections after it; append those (same file, under this heading) so the reader
        # doesn't mistake an empty-looking section for a missing one.
        nxt = chunk.id + 1
        while (len(_body(raw)) < MIN_BODY_CHARS and nxt < len(self.chunks) and len(raw) < max_chars
               and self.chunks[nxt].file == chunk.file and chunk.path and self.chunks[nxt].path.startswith(chunk.path)):
            raw += "\n\n" + PAGE_MARKER.sub("", self.index.chunk_text(self.chunks[nxt]))
            nxt += 1
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(line.rstrip() for line in raw.splitlines())).strip()
        return text if len(text) <= max_chars else text[:max_chars].rstrip() + " […passage continues]"

    # --- internals ------------------------------------------------------------------------

    def _passage(self, c: Chunk, score: float, query: str, at: int | None = None) -> Passage:
        """Chunk -> Passage: copies its place and adds the score and a snippet.
        _passage(chunk 143, 9.22, "spill contingency plan") -> Passage(143, "body.md", "5.3", …, score=9.22, snippet=…)"""
        return Passage(c.id, c.file, c.section, c.heading, c.pages, float(score),
                       self._snippet(c, query, at))

    def _snippet(self, c: Chunk, query: str, at: int | None) -> str:
        """About SNIPPET_CHARS of the chunk, markers removed and whitespace collapsed, centred on the
        grep match (at = its offset) or on the first query word (4+ letters) found; "…" marks cuts.
        A chunk shorter than SNIPPET_CHARS is returned whole.
        _snippet(chunk 143, "spill contingency plan", None) -> "### 5.3 Spill Contingency Plan The Spill Contingency Plan (Appendix E) was devel…" """
        raw = self.index.chunk_text(c)
        text = " ".join(PAGE_MARKER.sub(" ", raw).split())
        if len(text) <= SNIPPET_CHARS:
            return text
        if at is not None:
            pos = len(" ".join(PAGE_MARKER.sub(" ", raw[:at - c.start]).split()))
        else:
            low = text.lower()
            hits = [low.find(w) for w in re.findall(r"[a-z0-9]{4,}", query.lower()) if low.find(w) >= 0]
            pos = min(hits) if hits else 0
        start = max(0, min(pos - SNIPPET_CHARS // 3, len(text) - SNIPPET_CHARS))
        return ("…" if start else "") + text[start:start + SNIPPET_CHARS] + ("…" if start + SNIPPET_CHARS < len(text) else "")


MIN_BODY_CHARS = 200     # a chunk with less text than this under its headings is "just a heading"


def _body(text: str) -> str:
    """The text without heading lines: _body("#### 2.3.6 Wildlife\\n\\n") -> "" """
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#")).strip()


def phrase_pattern(phrase: str) -> re.Pattern | None:
    """A regex over normalized text for an exact phrase. Thousands separators are optional
    ("5,252" = "5252") and a number never matches inside a longer number.
    phrase_pattern("5,252")    -> re.compile(r"(?<![\\d.,])5,?2,?5,?2(?![\\d]|[.,]\\d)")
        matches "elevation 5,252 feet" and "5252", not "15,252" or "5,2521"
    phrase_pattern("2.7 M CY") -> matches "2.7 m cy", not "32.7 m cy"
    phrase_pattern("a")        -> None  (too short)"""
    p = normalize(phrase)
    if len(p) < 2:
        return None
    p = re.sub(r"(?<=\d),(?=\d{3}\b)", "", p)            # "5,252" -> "5252"
    body = re.escape(p)
    body = re.sub(r"(\d)(?=\d)", r"\1,?", body)          # allow "5,252" in the text for "5252"
    if p[0].isdigit():
        body = r"(?<![\d.,])" + body
    if p[-1].isdigit():
        body = body + r"(?![\d]|[.,]\d)"
    return re.compile(body)


def _appendix_letter(file: str) -> str | None:
    """"appendix_E.md" -> "E";  "appendix_B_summary.md" -> "B";  "body.md" -> None"""
    m = re.match(r"appendix_([A-Z])(?:_|\.)", file)
    return m.group(1) if m else None
