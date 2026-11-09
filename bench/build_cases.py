"""Build flawed test cases and the answer key from a gold set and flaw specs.

Usage:
    uv run python -m bench.build_cases thacker_pass

Reads   bench/labels/<plan>.yaml           clean-plan labels (item status, known gaps); a variant
                                           such as thacker_pass_clean sets `extends`, `gold`, `flaws`
        corpus/<plan>/gold/*.md            gold set (or the variant's `gold` directory)
        bench/flaws/<plan>/*.yaml          one flaw spec per case (format: bench/README.md)
        knowledge/checklist.yaml           checklist item ids
Writes  bench/cases/<plan>/<case>/*.md     the case's full document set (what the agent sees)
        bench/cases/<plan>/cases.json      list of case ids and files (no answers)
        bench/diffs/<plan>/<case>.patch    unified diff vs. gold, for human review only
        bench/answer_key/<plan>.json       answer key (never shown to the agent)

The build fails (exit 1, nothing written) if any spec is invalid: unknown type, difficulty or
checklist id; a `find` that does not match exactly once; a non-control case identical to gold;
or a missing/vague flaw that targets a checklist item not labeled `present` in the clean plan;
an inconsistent flaw without a `related` anchor; an implausible flaw without a `rationale`;
a missing, vague or implausible flaw without `absent` phrases, or with one that survives.

`absent:` (a list of phrases, or {text, files}) names the specifics a flaw removes or replaces:
the deleted content, the hedged-away details, or the old value of an implausible edit. Each
phrase must occur in the gold set and must not occur anywhere in the case (or in the listed
files), ignoring page markers and line breaks. This catches copies elsewhere in the plan that
the edits missed.

`related:` anchors ([{file, find, note}]) mark unedited text that a correct finding may cite,
such as the untouched value an inconsistent edit contradicts. Each must match exactly once in
the case text; it is stored in the answer key with role "related" and never edited.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import sys
from pathlib import Path

import yaml

from prep.patching import PatchError, apply_edit

ROOT = Path(__file__).resolve().parents[1]
TYPES = {"missing", "vague", "inconsistent", "implausible", "missed_trigger", "none"}
DIFFICULTIES = {"easy", "medium", "subtle", "none"}
PAGE_MARKER = re.compile(r"<!-- page (\d+)(?: \| printed ([^ ]+))? -->")
PAGE_MARKER_FULL = re.compile(r"<!-- page \d+(?: \| printed [^ ]+)? -->")
HEADING = re.compile(r"^(#{2,6}) (.+)$", re.M)


def locate(text: str, offset: int, find: str = "") -> dict:
    """PDF page and section heading for an edit at `offset` in a gold file. If the edited text
    starts with a heading (a deleted section), that heading is the location."""
    first_line_end = offset + len(find.lstrip("\n").split("\n", 1)[0]) + (len(find) - len(find.lstrip("\n")))
    page = printed = None
    for m in PAGE_MARKER.finditer(text, 0, max(offset, 0) + 1):
        page, printed = int(m.group(1)), m.group(2)
    heading = None
    for m in HEADING.finditer(text, 0, max(first_line_end, offset + 1)):
        heading = m.group(2).strip()
    in_front_matter = "....." in text[text.rfind("\n", 0, offset) + 1: text.find("\n", offset)]
    sec = re.match(r"^(\d+(?:\.\d+)*)\b", heading or "")
    return {
        "page": page,
        "printed_page": printed,
        "heading": "Table of Contents" if in_front_matter else heading,
        "section": None if in_front_matter else (sec.group(1) if sec else None),
        "role": "toc" if in_front_matter else "content",
    }


def load_labels(plan: str) -> dict:
    """Labels for a plan or variant. A variant names its base with `extends:`; it inherits the
    base's items (overridden one by one) and facts, and may set `gold:` (a gold directory) and
    `flaws:` (the plan whose flaw specs it reuses). `known_gaps` is never inherited."""
    labels = yaml.safe_load((ROOT / "bench" / "labels" / f"{plan}.yaml").read_text())
    base_id = labels.get("extends")
    if base_id:
        base = load_labels(base_id)
        items = {**base.get("items", {}), **(labels.get("items") or {})}
        labels = {**base, "known_gaps": [], **labels, "items": items}
        labels.setdefault("flaws", base.get("flaws", base_id))
    labels.setdefault("gold", f"corpus/{plan}/gold")
    labels.setdefault("flaws", plan)
    labels["known_gaps"] = labels.get("known_gaps") or []
    return labels


def load_specs(flaws: str) -> list[tuple[Path, dict]]:
    specs = []
    for p in sorted((ROOT / "bench" / "flaws" / flaws).glob("*.yaml")):
        specs.append((p, yaml.safe_load(p.read_text())))
    return specs


def validate_and_apply(spec: dict, path: Path, gold: dict[str, str], checklist_ids: set[str],
                       labels: dict) -> tuple[dict[str, str], list[dict], list[str]]:
    errors: list[str] = []
    where = path.name
    cid = spec.get("id")
    if cid != path.stem:
        errors.append(f"{where}: id {cid!r} must equal the file name {path.stem!r}")
    ftype, diff = spec.get("type"), spec.get("difficulty")
    if ftype not in TYPES:
        errors.append(f"{where}: unknown type {ftype!r}")
    if diff not in DIFFICULTIES:
        errors.append(f"{where}: unknown difficulty {diff!r}")
    is_control = ftype == "none"
    ids = spec.get("checklist_ids") or []
    if not is_control and not ids:
        errors.append(f"{where}: checklist_ids is empty")
    for i in ids:
        if i not in checklist_ids:
            errors.append(f"{where}: unknown checklist id {i!r}")
    if ftype in ("missing", "vague"):
        for i in ids:
            st = labels["items"].get(i, {}).get("status")
            if st != "present":
                errors.append(f"{where}: {ftype} flaw targets {i!r}, which is {st!r} in the clean plan (must be 'present')")
    for field in ("description", "expected_finding"):
        if not is_control and not spec.get(field):
            errors.append(f"{where}: `{field}` is required")
    if ftype == "inconsistent" and not spec.get("related"):
        errors.append(f"{where}: an inconsistent flaw needs a `related` anchor at the text it contradicts")
    if ftype == "implausible" and not spec.get("rationale"):
        errors.append(f"{where}: an implausible flaw needs a `rationale`")
    edits = spec.get("edits") or []
    if is_control and edits:
        errors.append(f"{where}: a control case (type: none) must have no edits")
    if not is_control and not edits:
        errors.append(f"{where}: no edits")

    files = dict(gold)
    locations: list[dict] = []
    for n, edit in enumerate(edits, 1):
        name = edit.get("file")
        if name not in files:
            errors.append(f"{where} edit #{n}: unknown file {name!r}")
            continue
        if edit.get("delete_file"):
            # the whole document is left out of the case (e.g. an appendix that was never attached)
            del files[name]
            locations.append({"file": name, "page": None, "printed_page": None, "heading": None,
                              "section": None, "role": "file",
                              "original": f"<entire file {name}>", "replacement": "<file removed>"})
            continue
        lost = [m for m in PAGE_MARKER_FULL.findall(edit.get("find") or "") if m not in (edit.get("replace") or "")]
        if lost:
            errors.append(f"{where} edit #{n} ({name}): deletes page marker(s) {lost}; keep them in `replace`")
        before = files[name]
        offset = before.find(edit.get("find", "")) if edit.get("find") else -1
        try:
            files[name] = apply_edit(before, edit, where=f"{where} edit #{n} ({name})")
        except PatchError as e:
            errors.append(str(e))
            continue
        loc = locate(before, offset, edit["find"])
        locations.append({
            "file": name, **loc,
            "original": edit["find"],
            "replacement": edit.get("replace", ""),
        })
    for n, rel in enumerate(spec.get("related") or [], 1):
        name, find = rel.get("file"), rel.get("find") or ""
        if name not in files:
            errors.append(f"{where} related #{n}: unknown file {name!r}")
            continue
        count = files[name].count(find) if find else 0
        if count != 1:
            errors.append(f"{where} related #{n} ({name}): `find` matched {count} times (must be exactly 1): {find[:77]!r}")
            continue
        loc = locate(files[name], files[name].find(find), find)
        locations.append({"file": name, **loc, "role": "related", "original": find, "replacement": None,
                          "note": rel.get("note", "")})
    if not errors and not is_control and files == gold:  # (a deleted file also counts as a change)
        errors.append(f"{where}: edits leave the documents unchanged")
    if ftype in NEEDS_ABSENT and not spec.get("absent"):
        errors.append(f"{where}: a {ftype} flaw needs `absent` phrases (the specifics that must not survive anywhere)")
    for n, entry in enumerate(spec.get("absent") or [], 1):
        text, only = (entry, None) if isinstance(entry, str) else (entry.get("text", ""), entry.get("files"))
        phrase = _flat(text)
        if not phrase:
            errors.append(f"{where} absent #{n}: empty phrase")
            continue
        if not any(phrase in _flat(t) for t in gold.values()):
            errors.append(f"{where} absent #{n}: {text[:60]!r} does not occur in the gold set, so it guards nothing")
        survivors = [f for f, t in files.items() if (only is None or f in only) and phrase in _flat(t)]
        if survivors:
            errors.append(f"{where} absent #{n}: {text[:60]!r} still occurs in {survivors}; a copy survived the edits")
    return files, locations, errors


NEEDS_ABSENT = {"missing", "vague", "implausible"}


def _flat(text: str) -> str:
    """Text with page markers removed and whitespace collapsed, so a phrase split by a page
    break or a line wrap still matches."""
    return " ".join(PAGE_MARKER_FULL.sub(" ", text).split())


def build(plan: str, out_root: Path = ROOT / "bench") -> dict:
    labels = load_labels(plan)
    gold_dir = ROOT / labels["gold"]
    gold = {p.name: p.read_text() for p in sorted(gold_dir.glob("*.md"))}
    if not gold:
        sys.exit(f"No gold files in {gold_dir}")
    checklist = yaml.safe_load((ROOT / "knowledge" / "checklist.yaml").read_text())
    checklist_ids = {i["id"] for i in checklist["items"]}

    specs = load_specs(labels["flaws"])
    if not specs:
        sys.exit(f"No flaw specs in bench/flaws/{labels['flaws']}/")

    results, all_errors = [], []
    for path, spec in specs:
        files, locations, errors = validate_and_apply(spec, path, gold, checklist_ids, labels)
        all_errors += errors
        results.append((spec, files, locations))
    if all_errors:
        print("Case build failed:\n  " + "\n  ".join(all_errors), file=sys.stderr)
        sys.exit(1)

    cases_dir = out_root / "cases" / plan
    diffs_dir = out_root / "diffs" / plan
    for d in (cases_dir, diffs_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    answer_cases, index = [], []
    for spec, files, locations in results:
        cid = spec["id"]
        cdir = cases_dir / cid
        cdir.mkdir()
        for name, text in files.items():
            (cdir / name).write_text(text, encoding="utf-8")
        index.append({"case_id": cid, "dir": f"bench/cases/{plan}/{cid}", "files": sorted(files)})

        patch = []
        for name in sorted(gold):
            if name not in files:
                patch.append(f"--- gold/{name}\n+++ {cid}/{name}\n@@ file removed from this case @@\n")
            elif files[name] != gold[name]:
                patch += difflib.unified_diff(
                    gold[name].splitlines(keepends=True), files[name].splitlines(keepends=True),
                    fromfile=f"gold/{name}", tofile=f"{cid}/{name}", n=1)
        (diffs_dir / f"{cid}.patch").write_text("".join(patch), encoding="utf-8")

        answer_cases.append({
            "case_id": cid,
            "type": spec["type"],
            "difficulty": spec["difficulty"],
            "checklist_ids": spec.get("checklist_ids") or [],
            "description": spec.get("description", ""),
            "expected_finding": spec.get("expected_finding", ""),
            "tells": spec.get("tells", ""),
            "rationale": spec.get("rationale", ""),
            "locations": locations,
        })

    (cases_dir / "cases.json").write_text(json.dumps({"plan": plan, "cases": index}, indent=1), encoding="utf-8")
    key = {
        "plan": plan,
        "checklist_version": checklist.get("version"),
        "known_gaps": labels.get("known_gaps", []),
        "cases": answer_cases,
    }
    key_dir = out_root / "answer_key"
    key_dir.mkdir(exist_ok=True)
    (key_dir / f"{plan}.json").write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    (key_dir / f"{plan}.md").write_text(review_sheet(key), encoding="utf-8")
    return key


def review_sheet(key: dict) -> str:
    """Human-readable table of all cases, for reviewing the answer key."""
    cell = lambda t: " ".join(str(t or "").split()).replace("|", "\\|")
    lines = [f"# Answer key: {key['plan']}", "",
             f"{len(key['cases'])} cases. Diffs: `bench/diffs/{key['plan']}/<case>.patch`.", "",
             "| Case | Type | Difficulty | Checklist | Where | Description | Tells / rationale |",
             "|---|---|---|---|---|---|---|"]
    for c in key["cases"]:
        where = ", ".join(sorted({f"{l['file']} §{l['section']}" if l.get("section") else f"{l['file']} p{l.get('page') or '-'}"
                                  for l in c["locations"] if l.get("role") != "related"}))
        extra = c.get("tells") or ""
        if c.get("rationale"):
            extra = (extra + " Rationale: " if extra else "Rationale: ") + c["rationale"]
        lines.append(f"| {c['case_id']} | {c['type']} | {c['difficulty']} | {', '.join(c['checklist_ids'])} | "
                     f"{cell(where)} | {cell(c['description'])} | {cell(extra)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", help="plan id, e.g. thacker_pass")
    args = ap.parse_args()
    key = build(args.plan)
    from collections import Counter
    c = key["cases"]
    print(f"Built {len(c)} cases for {args.plan}")
    print("  by type:      ", dict(Counter(x["type"] for x in c)))
    print("  by difficulty:", dict(Counter(x["difficulty"] for x in c)))


if __name__ == "__main__":
    main()
