"""Build a plan's gold set from its raw extraction plus reviewed manual fixes.

Usage:
    uv run python -m prep.build_gold corpus/thacker_pass

Reads   <plan>/gold_config.yaml   which output files to build and from which raw/handwritten inputs
        <plan>/raw/*.md           output of prep.extract_pdf (never edited by hand)
        <plan>/handwritten/*.md   files written by hand (summaries, figure descriptions)
        <plan>/fixes.yaml         manual corrections, as exact find/replace edits (see prep.patching)
Writes  <plan>/gold/*.md          the gold set (do not edit by hand; edit fixes.yaml and rebuild)

Order of operations per file: concatenate inputs -> automatic normalization -> manual fixes ->
strip navigation (TOC and lists of tables/figures/appendices) -> prepend header.

Navigation is stripped so the agent cannot find missing sections just by comparing the TOC with
the headings; it must read the plan. Page markers inside the stripped block are kept.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from prep.patching import PatchError, apply_edit

PAGE_MARKER = re.compile(r"<!-- page \d+(?: \| printed [^ ]+)? -->")
TERMINAL = (".", "!", "?", ":", ";", ")", "”", '"', "|")


def is_plain(par: str) -> bool:
    return bool(par) and not par.startswith(("#", "|", "<!--"))


def join_page_breaks(text: str) -> str:
    """Join a paragraph split by a page break: 'end of p.4 <!-- page 5 --> rest' becomes one
    paragraph with the marker kept inline, so quotes spanning the break still match."""
    paras = text.split("\n\n")
    out: list[str] = []
    i = 0
    while i < len(paras):
        p = paras[i]
        if (
            i + 2 < len(paras)
            and out
            and is_plain(out[-1].strip())
            and PAGE_MARKER.fullmatch(p.strip())
            and is_plain(paras[i + 1].strip())
            and not paras[i + 1].lstrip().startswith("- ")
            and not out[-1].rstrip().endswith(TERMINAL)
        ):
            out[-1] = out[-1].rstrip() + " " + p.strip() + " " + paras[i + 1].strip()
            i += 2
            continue
        out.append(p)
        i += 1
    return "\n\n".join(out)


def normalize(text: str) -> str:
    text = re.sub(r"^<!-- source: .*?-->\n", "", text)          # raw-file provenance line
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    text = join_page_breaks(text)
    # footnote paragraphs: "Note: 1 Text" -> "Note: [1] Text" (superscripts lose their size in extraction)
    text = re.sub(r"(?m)^Note: (\d) ", r"Note: [\1] ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def replace_pages(text: str, replacements: dict) -> str:
    """Replace the content of whole pages (e.g. scanned maps whose OCR text is noise) with a
    placeholder, keeping the page marker."""
    if not replacements:
        return text
    parts = re.split(r"(<!-- page \d+(?: \| printed [^ ]+)? -->)", text)
    seen = set()
    for i in range(1, len(parts), 2):
        n = int(re.search(r"\d+", parts[i]).group())
        seen.add(n)
        if n in replacements:
            parts[i + 1] = "\n\n" + replacements[n].strip() + "\n\n"
    missing = set(replacements) - seen
    if missing:
        raise SystemExit(f"replace_pages: no such pages {sorted(missing)}")
    return "".join(parts)


TOC_START = re.compile(r"(?m)^(?:#+ )?Table of Contents$")
NAV_HEADINGS = {"table of contents", "list of tables", "list of figures", "list of appendices",
                "list of attachments", "list of forms", "attachments"}
ANY_HEADING = re.compile(r"(?m)^#{1,6} (.+)$")


def strip_navigation(text: str) -> str:
    """Remove every table of contents together with the lists of tables, figures, appendices,
    attachments and forms that follow it, up to the next real heading. Page markers are kept."""
    pos = 0
    while (m := TOC_START.search(text, pos)):
        end = len(text)
        for h in ANY_HEADING.finditer(text, m.end()):
            if h.group(1).strip().lower() not in NAV_HEADINGS:
                end = h.start()
                break
        markers = PAGE_MARKER.findall(text[m.start():end])
        keep = "".join(mk + "\n\n" for mk in markers)
        text = text[:m.start()] + keep + text[end:]
        pos = m.start() + len(keep)
    return text


def header(spec: dict) -> str:
    lines = [f"<!-- gold file: {spec['out']} | {spec['source']} -->"]
    if spec.get("note"):
        lines.append(f"<!-- note: {spec['note']} -->")
    lines.append("")
    lines.append(f"# {spec['title']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def build(plan_dir: Path) -> dict[str, str]:
    cfg = yaml.safe_load((plan_dir / "gold_config.yaml").read_text())
    texts: dict[str, str] = {}
    for spec in cfg["files"]:
        if "handwritten" in spec:
            texts[spec["out"]] = (plan_dir / "handwritten" / spec["handwritten"]).read_text()
        else:
            parts = [(plan_dir / "raw" / r).read_text() for r in spec["raw"]]
            text = normalize("\n\n".join(parts))
            text = replace_pages(text, spec.get("replace_pages") or {})
            if spec.get("prepend"):
                text = spec["prepend"].strip() + "\n\n" + text
            texts[spec["out"]] = text

    fixes_path = plan_dir / "fixes.yaml"
    fixes = yaml.safe_load(fixes_path.read_text()) if fixes_path.exists() else None
    errors = []
    for i, edit in enumerate(fixes or []):
        try:
            texts[edit["file"]] = apply_edit(texts[edit["file"]], edit, where=f"fixes.yaml #{i + 1} ({edit['file']})")
        except (PatchError, KeyError) as e:
            errors.append(str(e))
    if errors:
        raise SystemExit("Gold build failed:\n  " + "\n  ".join(errors))

    out_dir = plan_dir / "gold"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.md"):
        old.unlink()
    for spec in cfg["files"]:
        body = texts[spec["out"]] if "handwritten" in spec else strip_navigation(texts[spec["out"]])
        full = body if "handwritten" in spec else header(spec) + body
        texts[spec["out"]] = full
        (out_dir / spec["out"]).write_text(full, encoding="utf-8")
    return texts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan_dir", type=Path)
    args = ap.parse_args()
    texts = build(args.plan_dir)
    for name, t in texts.items():
        print(f"{name:32s} {len(t):>9,} chars  {len(PAGE_MARKER.findall(t)):>4} page markers")


if __name__ == "__main__":
    main()
