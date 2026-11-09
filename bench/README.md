# Test bench

The test bench turns a clean plan of operations into test cases with known, planted flaws, then
scores any system's reports against the answer key.

```
corpus/<plan>/gold/*.md   ──┐
bench/flaws/<plan>/*.yaml ──┼─► bench.build_cases ─► bench/cases/<plan>/<case>/*.md   (agent input)
bench/labels/<plan>.yaml  ──┤                       bench/answer_key/<plan>.json     (hidden)
knowledge/checklist.yaml  ──┘                       bench/diffs/<plan>/<case>.patch  (review)

runs/<run>/<case>.json  ──► bench.score ─► runs/<run>/scores.{json,md}
```

## Commands

```bash
uv run python -m prep.build_gold corpus/thacker_pass     # rebuild the gold set (after editing fixes.yaml)
uv run python -m bench.build_cases thacker_pass          # rebuild cases + answer key (after editing flaws)
uv run python -m bench.score thacker_pass_clean --oracle    # self-check: must be 1.0
uv run python -m bench.score thacker_pass_clean runs/<run>  # score a system's reports
uv run pytest -q
```

**The scorer only supports plans with no known gaps** (for now, `thacker_pass_clean`). It ignores
`known_gaps` entirely, and the command line stops with an `AssertionError` if the answer key has any.

## Clean development variant (`thacker_pass_clean`)

The published plan has real deficiencies of its own (the `known_gaps` in the labels). For building
and debugging an agent there is a second variant in which every known gap is repaired:

```bash
uv run python -m prep.build_clean corpus/thacker_pass    # gold/ + repairs.yaml -> gold_clean/
uv run python -m bench.build_cases thacker_pass_clean    # same 40 flaw specs, planted on gold_clean/
uv run python -m bench.score thacker_pass_clean --oracle
uv run python -m bench.score thacker_pass_clean runs/<run>
```

- **What it is.** `corpus/thacker_pass/repairs.yaml` holds one find/replace repair per known gap. Most
  align an inconsistent value or reference. A few add plausible plan text where the plan is silent
  (riparian mitigation, the chance-find procedure, the water balance, Phase 2 water rights, closure
  stabilization, pit backfill factors). Each flawed case then contains exactly one deficiency, and the
  control TP-000 should get **zero** flags.
- **Dev only.** The repair text was written for this project, not by the operator. Official numbers
  belong on `thacker_pass`, but the scorer can't score that plan until known-gap support is
  added back.
- **Configuration.** [labels/thacker_pass_clean.yaml](labels/thacker_pass_clean.yaml) `extends` the
  main labels, points `gold` at `gold_clean/` and `flaws` at the shared specs, overrides the four
  repaired items to `present`, and has no known gaps.
- **Shared specs.** Every spec must build on both variants. Repairs reuse the plan's own values,
  never repeat a value a flaw edits, and add no monitoring frequencies (TP-012, TP-018). The
  exactly-once anchors and the `absent` guard catch most conflicts on the clean build.
- **Only known gaps are repaired.** A confident false alarm on the clean TP-000 may be a real gap
  nobody listed. Check it against the unrepaired plan and, if it is real, add it to `known_gaps`
  and a repair here.

## Rules

- **The agent sees only `bench/cases/<plan>/<case>/`.** These folders hold nothing but plan documents.
  Never give the agent `bench/flaws/`, `bench/answer_key/`, `bench/diffs/` or `bench/labels/`.
- **There is one planted flaw per case.** `TP-000` is the clean control, used to count false alarms.
- **Missing and vague flaws may only target checklist items labeled `present`** in
  `bench/labels/<plan>.yaml`. The builder enforces this.
- **No copy may survive.** Missing, vague and implausible specs list `absent` phrases, and the builder
  rejects a case in which any of them still occurs anywhere in the document set.
- **Real gaps in the published plan** are listed as `known_gaps` in the labels. The scorer doesn't
  use them yet (see above), so only plans with none can be scored.

## Flaw spec format (`bench/flaws/<plan>/TP-###.yaml`)

```yaml
id: TP-014                      # must equal the file name
type: inconsistent              # missing | vague | inconsistent | implausible | missed_trigger | none
difficulty: subtle              # easy | medium | subtle | none
checklist_ids: [ops-designs]    # ids from knowledge/checklist.yaml
description: What was changed (for humans).
expected_finding: What a good reviewer should report.
tells: Clues left in the document that make the flaw easier to find (sets the difficulty).
rationale: Why the value is implausible, with a reference value (required for implausible).
edits:                          # applied in order to the gold set
  - file: body.md
    find: "exact text, must occur exactly once"
    replace: "new text"         # "" deletes; page markers in `find` must be kept in `replace`
  - file: appendix_G.md
    delete_file: true           # leave the whole document out of the case
related:                        # unedited text a correct finding may cite (required for inconsistent)
  - file: body.md
    find: "the untouched value the edit contradicts"
    note: "§6.5.2"
absent:                         # required for missing, vague, implausible
  - "NAC 534"                   # must occur in gold, and nowhere in the case
  - {text: "Year 5", files: [body.md]}   # optionally checked only in some files
```

`absent` phrases name the specifics the flaw removes: the deleted content, the hedged-away details,
or the old value of an implausible edit. The check ignores page markers and line breaks. Pick
distinctive phrases and search the whole gold set for them before writing the edits: the same
detail is often repeated in another section or an appendix.

**Authoring rules per type**

- **missing:** delete the content and every copy of it elsewhere in the body or appendices, so the
  item is truly absent. Passing mentions may stay only if they carry none of the specifics.
- **vague:** keep the topic but replace the specifics (numbers, methods, schedules) with hedges such as
  "as appropriate" or "to be determined", everywhere they appear. Use the checklist item's `vague_if`
  as the test.
- **inconsistent:** change **one** occurrence of a value that appears in several places, to a value
  that is plausible on its own. Add a `related` anchor at an untouched occurrence.
- **implausible:** change **every** occurrence to a value an expert would reject, so there is no
  contradiction to find. Explain why in `rationale`.
- **missed_trigger** (not yet used): add a fact that triggers a requirement, and nothing that meets it.
- Never delete page markers (the builder rejects it) and avoid the clean plan's known gaps.

The builder works out the page and section of every edit from the gold page markers and
headings. You don't type locations.

**Difficulty guide**

- **easy:** obvious tells remain. Examples: a whole appendix is missing but still cited, or a deleted
  section is still referenced elsewhere ("as described in Section 6.5.16").
- **medium:** the direct tells are cleaned up, such as cross-references. Indirect clues remain, such as
  a gap in section numbering.
- **subtle:** a single value, sentence or qualifier is changed. Finding it needs cross-reading or
  domain knowledge.

The gold set has **no tables of contents** (or lists of tables, figures or appendices). `prep.build_gold`
strips them so an agent can't find missing sections just by comparing the TOC with the headings.
Their page markers are kept, so page numbers still match the PDF.

## Report format

See `schema/report.schema.json`. Write one `<case_id>.json` per case. Each flag has a
`checklist_id`, a `type`, an `explanation`, and optionally `file`, `section`, `pages` and `quote`.

A flag catches the planted flaw if it names the right checklist item **at the right place**, so give
a `section` or `pages`, including for missing content (cite the section where it belongs). The one
exception is a whole missing document, where the right checklist item alone is enough. The flag's
`type` is not scored. The matching rule is described at the top of `score.py`.
