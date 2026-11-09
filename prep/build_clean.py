"""Build the repaired, development-only gold set: gold/ plus repairs.yaml -> gold_clean/.

Usage:
    uv run python -m prep.build_clean corpus/thacker_pass

Reads   <plan>/gold/*.md          the gold set (built by prep.build_gold)
        <plan>/repairs.yaml       one repair per known gap, as exact find/replace edits
Writes  <plan>/gold_clean/*.md    the gold set with every known gap repaired

The repaired set lets an agent be developed against a plan whose only deficiency is the planted
one (bench plan id `<plan>_clean`). Its repair text is written for this project, not by the
operator, so official scores always use the unrepaired gold set.

The build fails (nothing written) if a repair's `find` does not match exactly once, or if a
repair would delete a page marker. File contents are otherwise byte-identical to gold.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import yaml

from prep.patching import PatchError, apply_edit

PAGE_MARKER = re.compile(r"<!-- page \d+(?: \| printed [^ ]+)? -->")


def build(plan_dir: Path) -> tuple[dict[str, str], list[dict]]:
    gold = {p.name: p.read_text() for p in sorted((plan_dir / "gold").glob("*.md"))}
    repairs = yaml.safe_load((plan_dir / "repairs.yaml").read_text()) or []
    files, errors = dict(gold), []
    for n, r in enumerate(repairs, 1):
        where = f"repairs.yaml #{n} ({r.get('id')}, {r.get('file')})"
        if r.get("file") not in files:
            errors.append(f"{where}: unknown file")
            continue
        lost = [m for m in PAGE_MARKER.findall(r.get("find") or "") if m not in (r.get("replace") or "")]
        if lost:
            errors.append(f"{where}: deletes page marker(s) {lost}")
            continue
        try:
            files[r["file"]] = apply_edit(files[r["file"]], r, where=where)
        except PatchError as e:
            errors.append(str(e))
    if errors:
        raise PatchError("\n  ".join(["repairs failed:"] + errors))
    return files, repairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan_dir", type=Path, help="e.g. corpus/thacker_pass")
    args = ap.parse_args()
    try:
        files, repairs = build(args.plan_dir)
    except PatchError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    out = args.plan_dir / "gold_clean"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    for name, text in files.items():
        (out / name).write_text(text)
    gaps = sorted({r["gap"] for r in repairs})
    print(f"Applied {len(repairs)} repairs ({len(gaps)} known gaps) -> {out}")


if __name__ == "__main__":
    main()
