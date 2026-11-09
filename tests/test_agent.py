"""Tests for the review agent (agent/). Offline: no network, no LLM.

Run:  uv run pytest -q
"""

import json
import re
from pathlib import Path

import pytest

from agent.plan import PAGE_MARKER, Location, PlanIndex, normalize

ROOT = Path(__file__).resolve().parents[1]
PLAN = "thacker_pass_clean"
CASES = ROOT / "bench" / "cases" / PLAN


@pytest.fixture(scope="module")
def tp000() -> PlanIndex:
    return PlanIndex.load(CASES / "TP-000")


# ----------------------------------------------------------------- 0.1 PlanIndex: agreement
def _key_spots():
    """Every answer-key spot whose text can be found exactly once in its case: the replacement
    text of an edit (content) or the untouched anchor text (related)."""
    key = json.loads((ROOT / "bench" / "answer_key" / f"{PLAN}.json").read_text())
    for case in key["cases"]:
        for loc in case["locations"]:
            if loc["role"] == "content":
                text = loc.get("replacement") or ""
            elif loc["role"] == "related":
                text = loc.get("original") or ""
            else:
                continue                                  # whole deleted files, TOC rows
            text = text.strip()
            # A deletion's replacement is often just the page marker it had to keep: that marker
            # starts the *next* page, so it isn't a spot to check.
            if len(PAGE_MARKER.sub("", text).strip()) >= 8:
                yield case["case_id"], loc, text


def test_index_agrees_with_the_bench_on_every_answer_key_spot():
    """PlanIndex.locate gives the page (and section, where the key has one) that
    bench/build_cases.py: locate() recorded for every findable answer-key spot."""
    indexes, checked, skipped, problems = {}, 0, 0, []
    for case_id, loc, text in _key_spots():
        if case_id not in indexes:
            indexes[case_id] = PlanIndex.load(CASES / case_id)
        ix = indexes[case_id]
        raw = ix.text(loc["file"])
        if raw.count(text) != 1:
            skipped += 1                                  # ambiguous or rewritten across a marker
            continue
        start = raw.index(text)
        got = ix.locate(loc["file"], start, start + len(text))
        checked += 1
        if loc["page"] is not None and (not got.pages or got.pages[0] != loc["page"]):
            problems.append(f"{case_id} {loc['file']} page: key {loc['page']}, index {got.pages}")
        if loc["section"] is not None and got.section != loc["section"]:
            problems.append(f"{case_id} {loc['file']} section: key {loc['section']}, index {got.section}")
    assert checked >= 60, f"only {checked} spots could be checked ({skipped} skipped)"
    assert not problems, "\n".join(problems)


# ----------------------------------------------------------------- 0.1 PlanIndex: structure
def test_files_are_the_listed_ones_with_body_first(tp000):
    assert tp000.files[0] == "body.md"
    assert len(tp000.files) == 12


def test_cases_json_listing_is_respected():
    """TP-001 leaves out Appendix E, so the index must not have it."""
    ix = PlanIndex.load(CASES / "TP-001")
    assert "appendix_E.md" not in ix.files
    assert "body.md" in ix.files


def test_chunks_cover_all_real_text(tp000):
    """Chunks don't overlap, and everything they skip is whitespace or page markers."""
    for file in tp000.files:
        text = tp000.text(file)
        chunks = sorted((c for c in tp000.chunks if c.file == file), key=lambda c: c.start)
        pos = 0
        for c in chunks:
            assert c.start >= pos, f"{file}: chunks overlap at {c.start}"
            gap = PAGE_MARKER.sub("", text[pos:c.start])
            assert not gap.strip(), f"{file}: text not in any chunk at {pos}: {gap[:60]!r}"
            pos = c.end
        assert not PAGE_MARKER.sub("", text[pos:]).strip(), f"{file}: text after the last chunk"


def test_chunks_are_labelled_and_bite_sized(tp000):
    ids = [c.id for c in tp000.chunks]
    assert ids == list(range(len(ids)))
    body = [c for c in tp000.chunks if c.file == "body.md"]
    assert all(c.pages for c in body if c.start > tp000.text("body.md").find("<!-- page"))
    assert sum(c.section is not None for c in body) > 0.8 * len(body)
    # long spans are split at page breaks; only a single oversized page may exceed the limit by much
    assert max(c.end - c.start for c in tp000.chunks) < 3 * 4000


def test_unnumbered_heading_takes_its_numbered_parents_section(tp000):
    outline = tp000.outline("body.md")
    nested = [h for h in outline if h.number is None and h.section is not None]
    assert nested, "expected unnumbered sub-headings inside numbered sections"
    for h in nested:
        parents = [p for p in outline if p.offset < h.offset and p.number is not None]
        assert h.section == parents[-1].section or h.section in {p.number for p in parents}


def test_headings_with_no_numbered_parent_have_no_section(tp000):
    summary = next(h for h in tp000.outline("body.md") if h.title == "Summary")
    assert summary.section is None


# ----------------------------------------------------------------- 0.1 PlanIndex: finding quotes
def test_find_and_locate_the_tp021_value():
    ix = PlanIndex.load(CASES / "TP-021")
    [m] = ix.find("| West WRSF | 37.2 |")
    assert ix.locate(m.file, m.start, m.end) == Location("body.md", "3.5", (51,))


def test_a_span_across_a_heading_gets_the_section_holding_most_of_it():
    """The '### 3.5 Waste Rock Storage Facilities' heading starts at offset 89150 of body.md."""
    ix = PlanIndex.load(CASES / "TP-021")
    assert ix.text("body.md")[89150:89159] == "### 3.5 W"
    assert ix.locate("body.md", 89100, 89400).section == "3.5"      # 50 chars in 3.4.3, 250 in 3.5
    assert ix.locate("body.md", 89000, 89200).section == "3.4.3"    # 150 in 3.4.3, 50 in 3.5
    assert ix.locate("body.md", 89050, 89250).section == "3.4.3"    # a tie goes to where it starts


def test_quote_across_an_inline_page_marker(tp000):
    """body.md has '... after which, <!-- page 7 | printed ii --> reclamation, ...' mid-sentence."""
    matches = tp000.find("after which, reclamation, site closure activities, and post-closure")
    assert matches
    m = matches[0]
    assert tp000.locate(m.file, m.start, m.end).pages == (6, 7)


def test_find_ignores_case_whitespace_and_quote_styles(tp000):
    raw = tp000.text("body.md")
    line = next(l for l in raw.splitlines() if re.search(r"[‘’“”–—]", l) and len(l) > 60)
    plain = line.translate(str.maketrans("‘’“”–—", "''\"\"--")).upper()
    plain = "  \n".join(plain.split(" "))                # mangle the whitespace too
    assert any(m.file == "body.md" for m in tp000.find(plain))


def test_find_returns_original_offsets(tp000):
    [m, *_] = tp000.find("| West WRSF | 32.7 |")
    assert normalize(tp000.text(m.file)[m.start:m.end]) == normalize("| West WRSF | 32.7 |")


def test_find_rejects_tiny_or_absent_quotes(tp000):
    assert tp000.find("the") == []
    assert tp000.find("this sentence is certainly not in the plan anywhere") == []


def test_section_and_page_helpers(tp000):
    assert tp000.section_exists("body.md", "3.5")
    assert not tp000.section_exists("body.md", "99.9")
    assert tp000.section_location("body.md", "3.5") == Location("body.md", "3.5", (51,))
    assert tp000.pages_exist("body.md", [51, 52])
    assert not tp000.pages_exist("body.md", [9999])


# ----------------------------------------------------------------- 0.2 search
@pytest.fixture(scope="module")
def search000(tp000):
    from agent.search import PlanSearch
    return PlanSearch(tp000)


def test_bm25_finds_the_spill_plan(search000):
    files = {p.file for p in search000.search("spill contingency plan", k=3)}
    assert "appendix_E.md" in files


def test_passages_carry_their_place(search000):
    p = search000.search("waste rock storage facility capacity", k=1)[0]
    assert p.file and p.pages and p.header().startswith(p.file)


def test_grep_ignores_thousands_separators_but_not_longer_numbers(search000):
    with_comma = {p.chunk_id for p in search000.grep("5,252")}
    without = {p.chunk_id for p in search000.grep("5252")}
    assert with_comma and with_comma == without
    assert search000.grep("32.7 M CY") and all("32.7" in p.snippet for p in search000.grep("32.7 M CY"))
    from agent.search import phrase_pattern
    pattern = phrase_pattern("2.7 M CY")
    assert pattern.search("capacity of 2.7 m cy") and not pattern.search("capacity of 32.7 m cy")
    assert phrase_pattern("5252").search("elevation 5,252 feet") and not phrase_pattern("525").search("5,252")


def test_missing_refs_find_a_deleted_appendix_and_section():
    from agent.search import PlanSearch
    tp001 = PlanSearch(PlanIndex.load(CASES / "TP-001"))     # Appendix E deleted
    assert any(r.target == "appendix_E.md" for r in tp001.missing_refs())
    tp002 = PlanSearch(PlanIndex.load(CASES / "TP-002"))     # body §6.5.16 deleted
    assert any(r.target == "6.5.16" for r in tp002.missing_refs())


def test_clean_plan_has_almost_no_missing_refs(search000):
    """Only the published plan's own broken reference (appendix C cites a §1.5.4 it doesn't have)."""
    assert [r.target for r in search000.missing_refs()] == ["1.5.4"]


def test_hybrid_merges_queries_without_repeats(search000):
    out = search000.hybrid(["riparian springs mitigation", '"seasonal springs"'], k=6)
    assert len(out) == len({p.chunk_id for p in out}) <= 6
    assert out[0].score >= out[-1].score


def test_tools_are_dspy_tools():
    from agent.tools import make_tools
    tools = make_tools(CASES / "TP-002")
    assert set(tools) == {"search_plan", "grep", "read_section", "outline", "missing_references"}
    assert "6.5.16" in tools["missing_references"]()
    assert "3.5 Waste Rock" in tools["read_section"](file="body.md", section="3.5")


# ----------------------------------------------------------------- 0.3 locator
def _draft(**kw):
    from agent.report import FlagDraft
    return FlagDraft(**{"checklist_id": "ops-designs", "type": "inconsistent", **kw})


def test_locator_replaces_guessed_place_with_the_quotes_place():
    from agent.anchor import anchor
    ix = PlanIndex.load(CASES / "TP-021")
    p = anchor(_draft(quote="| West WRSF | 37.2 | 5,252 |", section="3.4", pages=[50]), ix)
    assert p.how == "quote" and p.location == Location("body.md", "3.5", (51,))
    assert p.model_location == Location(None, "3.4", (50,))


def test_locator_uses_the_longest_fragment_of_a_shortened_quote(tp000):
    from agent.anchor import anchor
    p = anchor(_draft(quote="The West WRSF is designed ... with a storage capacity of approximately 32.7 M CY"), tp000)
    assert p.how == "quote" and p.location.section == "3.5"


def test_locator_prefers_the_named_file_when_a_quote_repeats(tp000):
    from agent.anchor import anchor
    quote = "after which, reclamation, site closure activities"
    assert {m.file for m in tp000.find(quote)} >= {"body.md", "appendix_C.md"}
    assert anchor(_draft(quote=quote, file="appendix_C.md"), tp000).location.file == "appendix_C.md"


def test_locator_fallback_ladder(tp000):
    from agent.anchor import anchor
    sec = anchor(_draft(type="missing", quote="text that is not in the plan at all", file="body.md", section="6.8"), tp000)
    assert sec.how == "section" and sec.location == Location("body.md", "6.8", (120,))
    model = anchor(_draft(file="body.md", section="99.9", pages=[51]), tp000)
    assert model.how == "model" and model.location == Location("body.md", None, (51,))
    none = anchor(_draft(file="appendix_E.md"), tp000)
    assert none.how == "none" and none.location == Location("appendix_E.md")
    assert anchor(_draft(), tp000).location is None


# ----------------------------------------------------------------- 0.4 checklist + reports
def test_checklist_loads_all_51_items():
    from agent.checklist import load_checklist
    items = load_checklist()
    assert len(items) == 51 and len({i.id for i in items}) == 51
    assert "Too vague if" in items[0].render() and items[0].query()


def test_report_is_valid_and_cleans_up(tp000):
    from agent.anchor import anchor
    from agent.report import to_report
    good = anchor(_draft(quote="| West WRSF | 32.7 | 5,252 |", explanation="x"), tp000)
    unknown = anchor(_draft(checklist_id="not-an-item", quote="| West WRSF | 32.7 | 5,252 |"), tp000)
    report = to_report("TP-000", "test", [good, good, unknown])
    assert [f["checklist_id"] for f in report["flags"]] == ["ops-designs", "other"]
    assert report["flags"][1]["raw_checklist_id"] == "not-an-item"
    assert report["flags"][0]["anchor"] == "quote"


def test_report_rejects_a_bad_report():
    import jsonschema
    from agent.report import validate
    with pytest.raises(jsonschema.ValidationError):
        validate({"case_id": "TP-000", "flags": [{"checklist_id": "x", "type": "wrong-type", "explanation": ""}]})


# ----------------------------------------------------------------- the wall
def test_index_refuses_folders_outside_bench_cases(tmp_path):
    (tmp_path / "x.md").write_text("# hi")
    with pytest.raises(ValueError):
        PlanIndex.load(tmp_path)
    with pytest.raises(ValueError):
        PlanIndex.load(ROOT / "bench" / "answer_key")


FORBIDDEN = re.compile(r"answer_key|bench/flaws|bench/labels|bench/diffs|from bench|import bench")


def test_agent_code_never_touches_the_hidden_side():
    offenders = [str(p.relative_to(ROOT)) for p in (ROOT / "agent").rglob("*.py")
                 if FORBIDDEN.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"agent/ must not reference the answer key, flaws, labels, diffs or bench code: {offenders}"
