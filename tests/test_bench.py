"""Tests for the patching module, the case builder and the scorer.

Run:  uv run pytest -q
"""

import json
from pathlib import Path

import pytest
import yaml

from bench import build_cases, score
from prep.patching import PatchError, apply_edit, apply_edits

ROOT = Path(__file__).resolve().parents[1]
PLAN = "thacker_pass"


# ------------------------------------------------------------------ patching
def test_apply_edit_replaces_exactly_once():
    assert apply_edit("a b c", {"find": "b", "replace": "X"}) == "a X c"


@pytest.mark.parametrize("text", ["a c", "b b"])
def test_apply_edit_rejects_zero_or_many_matches(text):
    with pytest.raises(PatchError):
        apply_edit(text, {"find": "b", "replace": "X"})


def test_apply_edits_unknown_file():
    with pytest.raises(PatchError):
        apply_edits({"a.md": "x"}, [{"file": "b.md", "find": "x", "replace": "y"}])


def test_gold_set_has_no_table_of_contents():
    for p in (ROOT / "corpus" / PLAN / "gold").glob("*.md"):
        t = p.read_text()
        assert "....." not in t and "Table of Contents" not in t, p.name


def test_strip_navigation_keeps_page_markers():
    from prep.build_gold import strip_navigation
    t = ("Intro.\n\n<!-- page 9 | printed iv -->\n\nTable of Contents\n\n1 Intro ..... 1 <!-- page 10 --> 2 B ..... 3\n\n"
         "## List of Tables\n\nTable 1 ..... 4\n\n## List of Acronyms\n\nBLM\n")
    out = strip_navigation(t)
    assert "....." not in out and "List of Tables" not in out and "## List of Acronyms" in out
    assert "<!-- page 9 | printed iv -->" in out and "<!-- page 10 -->" in out


# ------------------------------------------------------------------ builder validation
GOLD = {"body.md": "<!-- page 5 | printed 1 -->\n\n## 1 Intro\n\nAlpha beta.\n\n### 1.1 Sub\n\nGamma delta.\n"}
LABELS = {"items": {"op-tin": {"status": "present"}, "rec-riparian": {"status": "missing"}}}
IDS = {"op-tin", "rec-riparian"}


def _check(spec, name="TP-900"):
    path = Path(f"{name}.yaml")
    return build_cases.validate_and_apply(spec, path, GOLD, IDS, LABELS)


def _spec(**kw):
    base = {"id": "TP-900", "type": "vague", "difficulty": "easy", "checklist_ids": ["op-tin"],
            "description": "d", "expected_finding": "e",
            "edits": [{"file": "body.md", "find": "Gamma delta.", "replace": "Gamma."}],
            "absent": ["delta"]}
    base.update(kw)
    return base


def test_valid_spec_applies_and_locates():
    files, locs, errors = _check(_spec())
    assert errors == []
    assert "Gamma." in files["body.md"] and "Gamma delta." not in files["body.md"]
    assert locs[0]["page"] == 5 and locs[0]["section"] == "1.1"


def test_deleted_section_is_located_at_its_own_heading():
    spec = _spec(type="missing", edits=[{"file": "body.md", "find": "### 1.1 Sub\n\nGamma delta.\n", "replace": ""}])
    _, locs, errors = _check(spec)
    assert errors == [] and locs[0]["section"] == "1.1"


@pytest.mark.parametrize("override, fragment", [
    ({"type": "bogus"}, "unknown type"),
    ({"difficulty": "hard"}, "unknown difficulty"),
    ({"checklist_ids": ["nope"]}, "unknown checklist id"),
    ({"checklist_ids": ["rec-riparian"]}, "must be 'present'"),
    ({"edits": [{"file": "body.md", "find": "not there", "replace": ""}]}, "matched 0 times"),
    ({"edits": [{"file": "body.md", "find": "Gamma delta.", "replace": "Gamma delta."}]}, "unchanged"),
    ({"id": "TP-901"}, "must equal the file name"),
    ({"edits": [{"file": "body.md", "find": "<!-- page 5 | printed 1 -->\n\n## 1 Intro", "replace": "## 1 Intro"}]}, "deletes page marker"),
    ({"type": "inconsistent"}, "needs a `related` anchor"),
    ({"type": "implausible"}, "needs a `rationale`"),
    ({"absent": []}, "needs `absent` phrases"),
    ({"absent": ["Alpha"]}, "a copy survived"),
    ({"absent": ["zeta"]}, "does not occur in the gold set"),
])
def test_invalid_specs_are_rejected(override, fragment):
    _, _, errors = _check(_spec(**override))
    assert any(fragment in e for e in errors), errors


def test_absent_phrase_surviving_in_another_file_is_rejected():
    # the edit removes "delta" from the body, but an appendix still repeats it
    gold = dict(GOLD, **{"appendix_X.md": "<!-- page 9 -->\n\nSee Gamma\n<!-- page 10 -->\ndelta here.\n"})
    _, _, errors = build_cases.validate_and_apply(_spec(absent=["Gamma delta"]), Path("TP-900.yaml"), gold, IDS, LABELS)
    assert any("appendix_X.md" in e and "survived" in e for e in errors), errors  # matched across a page break
    _, _, errors = build_cases.validate_and_apply(
        _spec(absent=[{"text": "Gamma delta", "files": ["body.md"]}]), Path("TP-900.yaml"), gold, IDS, LABELS)
    assert errors == []  # scoped to the body only


def test_inconsistent_flaws_need_no_absent_phrases():
    spec = _spec(type="inconsistent", related=[{"file": "body.md", "find": "Alpha beta."}], absent=None)
    assert _check(spec)[2] == []


def test_related_anchor_is_located_but_not_edited():
    spec = _spec(type="inconsistent", related=[{"file": "body.md", "find": "Alpha beta.", "note": "n"}], absent=None)
    files, locs, errors = _check(spec)
    assert errors == [] and "Alpha beta." in files["body.md"]
    rel = [l for l in locs if l["role"] == "related"]
    assert len(rel) == 1 and rel[0]["section"] == "1"


def test_related_anchor_must_match_once():
    spec = _spec(type="inconsistent", related=[{"file": "body.md", "find": "nowhere"}])
    assert any("related #1" in e for e in _check(spec)[2])


def test_control_case_needs_no_edits():
    spec = {"id": "TP-900", "type": "none", "difficulty": "none", "checklist_ids": [], "edits": []}
    files, _, errors = _check(spec)
    assert errors == [] and files == GOLD


def test_delete_file_edit():
    gold = dict(GOLD, **{"appendix_X.md": "x"})
    spec = _spec(type="missing", edits=[{"file": "appendix_X.md", "delete_file": True}], absent=["x"])
    files, locs, errors = build_cases.validate_and_apply(spec, Path("TP-900.yaml"), gold, IDS, LABELS)
    assert errors == [] and "appendix_X.md" not in files and locs[0]["role"] == "file"


# ------------------------------------------------------------------ real build
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    key1 = build_cases.build(PLAN, out_root=out)
    key2 = build_cases.build(PLAN, out_root=out)
    return out, key1, key2


def test_real_build_is_deterministic(built):
    _, key1, key2 = built
    assert key1 == key2


def test_cases_hide_the_answers(built):
    out, key, _ = built
    for c in key["cases"]:
        cdir = out / "cases" / PLAN / c["case_id"]
        assert cdir.is_dir()
        assert all(p.suffix == ".md" for p in cdir.iterdir()), "only plan documents may be in a case folder"


def test_cases_keep_every_page_marker(built):
    import re
    marker = re.compile(r"<!-- page \d+(?: \| printed [^ ]+)? -->")
    out, key, _ = built
    gold = {p.name: marker.findall(p.read_text()) for p in (ROOT / "corpus" / PLAN / "gold").glob("*.md")}
    for c in key["cases"]:
        for p in (out / "cases" / PLAN / c["case_id"]).glob("*.md"):
            assert marker.findall(p.read_text()) == gold[p.name], (c["case_id"], p.name)


def test_inserted_text_has_no_telltale_words():
    for p in (ROOT / "bench" / "flaws" / PLAN).glob("*.yaml"):
        for e in yaml.safe_load(p.read_text()).get("edits") or []:
            r = (e.get("replace") or "").lower()
            assert not any(w in r for w in ("flaw", "tp-", "planted", "answer key")), p.name


def test_every_planted_flaw_has_a_location(built):
    _, key, _ = built
    for c in key["cases"]:
        if c["type"] != "none":
            assert c["locations"], c["case_id"]


def test_flaws_reference_real_checklist_items():
    ids = {i["id"] for i in yaml.safe_load((ROOT / "knowledge" / "checklist.yaml").read_text())["items"]}
    for p in (ROOT / "bench" / "flaws" / PLAN).glob("*.yaml"):
        spec = yaml.safe_load(p.read_text())
        assert set(spec.get("checklist_ids") or []) <= ids, p.name


# ------------------------------------------------------------------ scorer
# The scorer only supports plans with no known gaps, so its tests use the clean variant.
@pytest.fixture(scope="module")
def key(built):
    return built[1]


@pytest.fixture(scope="module")
def clean_key(tmp_path_factory):
    return build_cases.build("thacker_pass_clean", out_root=tmp_path_factory.mktemp("clean"))


def test_score_cli_refuses_a_plan_with_known_gaps(key, monkeypatch):
    assert key["known_gaps"]
    monkeypatch.setattr("sys.argv", ["bench.score", PLAN, "--oracle"])
    with pytest.raises(AssertionError, match="known gaps"):
        score.main()


def test_oracle_scores_perfectly(clean_key):
    s = score.score(clean_key, score.oracle_reports(clean_key))["summary"]
    assert s["recall"] == 1.0 and s["precision"] == 1.0 and s["false_alarms_total"] == 0


def test_empty_reports_score_zero_recall(clean_key):
    s = score.score(clean_key, {})["summary"]
    assert s["recall"] == 0.0 and s["precision"] is None and s["false_alarms_total"] == 0


def test_unrelated_flag_is_a_false_alarm_and_duplicates_are_neutral(clean_key):
    case = next(c for c in clean_key["cases"] if c["type"] != "none")
    good = score.oracle_reports(clean_key)[case["case_id"]][0]
    bogus = {"checklist_id": "pf-fire", "type": "missing", "file": "body.md", "pages": [1], "explanation": "x"}
    row = score.score_case(case, [good, dict(good), bogus])
    assert row["caught"] and row["duplicates"] == 1 and row["false_alarms"] == 1


def test_removed_document_is_caught_by_checklist_id_wherever_the_flag_points():
    # a case that leaves out figures.md: the agent can only cite the references to it
    case = {"case_id": "X", "type": "missing", "difficulty": "easy", "checklist_ids": ["ops-maps"],
            "locations": [{"file": "figures.md", "page": None, "section": None, "role": "file"}]}
    cites_body = {"checklist_id": "ops-maps", "type": "missing", "file": "body.md", "section": "3.3.1", "explanation": "x"}
    assert score.score_case(case, [cites_body])["caught"]
    no_location = {"checklist_id": "ops-maps", "type": "missing", "explanation": "x"}
    assert score.score_case(case, [no_location])["caught"]
    wrong_item = dict(cites_body, checklist_id="pf-fire")
    assert not score.score_case(case, [wrong_item])["caught"]


# --- the one rule: right checklist item at the right place (real cases from the clean key)
def _case(clean_key, case_id):
    return next(c for c in clean_key["cases"] if c["case_id"] == case_id)


def _caught(case, **flag):
    return score.score_case(case, [{"type": case["type"], "explanation": "x", **flag}])["caught"]


def test_right_item_needs_the_right_place(clean_key):
    tp031 = _case(clean_key, "TP-031")  # pf-exclusion at body.md §3.17, §3.24.6, §4.6
    assert _caught(tp031, checklist_id="pf-exclusion", file="body.md", section="3.17")
    assert _caught(tp031, checklist_id="pf-exclusion", file="body.md", section="4.6")    # any spot
    assert _caught(tp031, checklist_id="pf-exclusion", file="body.md", pages=[73])       # page ±1
    assert not _caught(tp031, checklist_id="pf-exclusion", file="body.md", section="9.9")
    assert not _caught(tp031, checklist_id="pf-exclusion", file="appendix_C.md", section="3.17")


def test_wrong_item_at_the_right_place_is_a_false_alarm(clean_key):
    row = score.score_case(_case(clean_key, "TP-031"), [
        {"checklist_id": "pf-wildlife-tes", "type": "implausible", "file": "body.md", "section": "3.17"}])
    assert not row["caught"] and row["false_alarms"] == 1


def test_flag_type_is_not_scored(clean_key):
    tp031 = _case(clean_key, "TP-031")
    assert _caught(tp031, checklist_id="pf-exclusion", type="inconsistent", file="body.md", section="3.17")


def test_a_flag_needs_a_section_or_pages(clean_key):
    tp031 = _case(clean_key, "TP-031")
    assert not _caught(tp031, checklist_id="pf-exclusion")                    # no location at all
    assert not _caught(tp031, checklist_id="pf-exclusion", file="body.md")    # a file but not where


def test_parent_section_of_a_deleted_section_counts(clean_key):
    tp010 = _case(clean_key, "TP-010")  # §6.4.8 Wildlife Habitat Restoration deleted
    assert _caught(tp010, checklist_id="rec-wildlife-habitat", file="body.md", section="6.4")


def test_any_of_a_cases_checklist_items_counts(clean_key):
    tp012 = _case(clean_key, "TP-012")  # mon-water or mon-details, at appendix_I §3.1.1 / §3.1.2
    assert _caught(tp012, checklist_id="mon-water", file="appendix_I.md", section="3.1.2")
    assert _caught(tp012, checklist_id="mon-details", file="appendix_I.md", section="3.1.1")


def test_every_flag_on_the_control_is_a_false_alarm(clean_key):
    control = _case(clean_key, "TP-000")
    oracle_flags = [f for flags in score.oracle_reports(clean_key).values() for f in flags]
    row = score.score_case(control, oracle_flags)
    assert not row["caught"] and row["false_alarms"] == len(oracle_flags) > 0


def test_score_cli_writes_outputs(tmp_path, clean_key, monkeypatch):
    for cid, flags in score.oracle_reports(clean_key).items():
        (tmp_path / f"{cid}.json").write_text(json.dumps({"case_id": cid, "flags": flags}))
    reports = score.load_reports(tmp_path, [c["case_id"] for c in clean_key["cases"]])
    assert score.score(clean_key, reports)["summary"]["recall"] == 1.0


# ------------------------------------------------------------------ clean (repaired) dev variant
from prep import build_clean  # noqa: E402

CLEAN = "thacker_pass_clean"
PLAN_DIR = ROOT / "corpus" / PLAN


def test_repairs_apply_and_cover_every_known_gap():
    files, repairs = build_clean.build(PLAN_DIR)
    active = {g["id"] for g in build_cases.load_labels(PLAN)["known_gaps"]}
    repaired = {r["gap"] for r in repairs}
    assert repaired <= active, f"repairs name unknown gaps: {repaired - active}"
    assert active <= repaired, f"known gaps without a repair: {active - repaired}"
    gold = {p.name: p.read_text() for p in (PLAN_DIR / "gold").glob("*.md")}
    for name, text in files.items():
        assert build_cases.PAGE_MARKER_FULL.findall(text) == build_cases.PAGE_MARKER_FULL.findall(gold[name]), name


def test_gold_clean_on_disk_is_current():
    files, _ = build_clean.build(PLAN_DIR)
    on_disk = {p.name: p.read_text() for p in (PLAN_DIR / "gold_clean").glob("*.md")}
    assert on_disk == files, "gold_clean/ is stale: run uv run python -m prep.build_clean corpus/thacker_pass"


def test_repair_text_has_no_telltale_words():
    # words that would reveal the edit, unless the published plan already uses them itself
    # (it does say "repaired as necessary", for example)
    _, repairs = build_clean.build(PLAN_DIR)
    gold_text = " ".join(p.read_text().lower() for p in (PLAN_DIR / "gold").glob("*.md"))
    for r in repairs:
        added = r["replace"].lower()
        for word in ("repair", "clean", "kg-", "known gap", "planted", "flaw", "deficien"):
            assert word not in added or word in gold_text, (r["id"], word)


def test_variant_labels_inherit_and_override():
    base, clean = build_cases.load_labels(PLAN), build_cases.load_labels(CLEAN)
    assert clean["known_gaps"] == [] and base["known_gaps"]
    assert clean["flaws"] == PLAN and clean["gold"] == "corpus/thacker_pass/gold_clean"
    assert set(clean["items"]) == set(base["items"])
    assert all(v["status"] == "present" for v in clean["items"].values())
    assert clean["facts"] == base["facts"]


def test_clean_variant_builds_the_same_cases_and_scores_perfectly(clean_key, key):
    assert clean_key["known_gaps"] == []
    assert [c["case_id"] for c in clean_key["cases"]] == [c["case_id"] for c in key["cases"]]
    s = score.score(clean_key, score.oracle_reports(clean_key))["summary"]
    assert s["recall"] == 1.0 and s["precision"] == 1.0
