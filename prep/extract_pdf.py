"""Extract a text-layer PDF to page-tagged Markdown using its layout.

Usage:
    uv run python -m prep.extract_pdf <input.pdf> <output.md> [--pages 5-131]

What it does, per page:
  * Drops running headers/footers (small text in the top/bottom margins) but records the
    printed page number from the footer.
  * Detects tables with PyMuPDF's table finder and emits them as Markdown tables.
  * Merges text fragments that sit on the same baseline (this restores section numbers that
    live in a separate OCR text layer, e.g. "3.5.1" + "Mine Waste Rock Characterization").
  * Rebuilds paragraphs from vertical spacing instead of emitting one paragraph per line.
  * Marks headings (medium/bold font) with Markdown '#' levels taken from their section number.
  * Normalizes bullet glyphs to "- ".

Each page starts with a marker:  <!-- page N | printed P -->   (N = 1-based PDF page index).
A per-page stats file <output>.stats.json is written for quality checks.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

HEADER_Y = 62          # lines ending above this are header zone
FOOTER_Y = 735         # lines starting below this are footer zone
SAME_LINE_DY = 3.5     # fragments whose vertical centers differ by less than this are one line
LOW_TEXT_THRESHOLD = 200

BULLET_GLYPHS = {"o", "•", "", "▪", "■", "➢", "Ø", "§", "◦", "-", "–", "", "", ""}
TOC_LINE = re.compile(r"\.{4,}\s*[0-9ivxlcA-Z-]+$")
LIST_START = re.compile(r"^(Figure \d+|Table \d+-\d+|Appendix [A-Z]\b|Attachment [A-Z0-9]\b)")
NUM_LABEL = re.compile(r"^\d+(\.\d+)*\.?$")
HEAVY_FONT = re.compile(r"(Medium|Bold|Demi|Heavy|Black|Semibold)", re.I)


@dataclass
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    spans: list = field(default_factory=list)  # (x0, text, font, size)

    @property
    def text(self) -> str:
        out = ""
        for _, t, _, _ in sorted(self.spans, key=lambda s: s[0]):
            if out and not out.endswith(" ") and not t.startswith(" "):
                out += " "
            out += t
        return re.sub(r"\s+", " ", out).strip()

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


def body_font(doc, pages) -> tuple[str, float]:
    c = collections.Counter()
    for i in pages:
        for b in doc[i].get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    c[(s["font"], round(s["size"]))] += len(s["text"].strip())
    (font, size), _ = c.most_common(1)[0]
    return font, size


def inside(bbox, rects, pad=2.0) -> bool:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return any(r.x0 - pad <= cx <= r.x1 + pad and r.y0 - pad <= cy <= r.y1 + pad for r in rects)


def table_markdown(tab) -> str:
    """Render a PyMuPDF table as Markdown, aligning cells by x-position.

    The table finder often splits one logical column into several grid columns (leaving
    empty spacer cells), and merged or blank cells shift values. So we take the column
    boundaries from the fullest row and place every non-empty cell in the logical column
    whose x-range contains the cell's center (nearest column if none does). Header rows
    that wrap onto following grid rows are merged back into the header.
    """
    values = tab.extract()
    if not values:
        return ""

    def clean(v):
        return re.sub(r"\s+", " ", (v or "").replace("|", "/")).strip()

    grid = []  # rows of [(x_center, x0, x1, text)]
    row_y1 = []
    for row, vals in zip(tab.rows, values):
        cells = []
        for bbox, v in zip(row.cells, vals):
            t = clean(v)
            if t and bbox is not None:
                cells.append(((bbox[0] + bbox[2]) / 2, bbox[0], bbox[2], t))
        if cells:
            grid.append(cells)
            row_y1.append(row.bbox[3])
    if not grid:
        return ""

    ref = max(grid, key=len)  # first fullest row defines the logical columns
    cols = [(c[1], c[2]) for c in ref]
    width = len(cols)

    def col_of(xc):
        for i, (x0, x1) in enumerate(cols):
            if x0 - 1 <= xc <= x1 + 1:
                return i
        return min(range(width), key=lambda i: abs((cols[i][0] + cols[i][1]) / 2 - xc))

    rows = []
    for cells in grid:
        r = [""] * width
        for xc, _, _, t in cells:
            i = col_of(xc)
            r[i] = (r[i] + " " + t).strip()
        rows.append(r)

    # merge wrapped header lines: a multi-line header is one tall grid row, and the table
    # finder adds sub-rows that lie inside its vertical extent
    header = rows[0]
    k = 1
    while k < len(rows) and row_y1[k] <= row_y1[0] + 1:
        header = [(h + " " + v).strip() for h, v in zip(header, rows[k])]
        k += 1
    rows = [header] + rows[k:]

    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def collect_lines(page, skip_rects) -> tuple[list[Line], str | None]:
    """Return content lines (merged by baseline) and the printed page number, if any."""
    raw: list[Line] = []
    printed = None
    footer_bits = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [(s["bbox"][0], s["text"], s["font"], s["size"]) for s in l["spans"] if s["text"].strip()]
            if not spans:
                continue
            x0, y0, x1, y1 = l["bbox"]
            if y1 < HEADER_Y:
                continue
            if y0 > FOOTER_Y:
                footer_bits.append((x0, " ".join(s[1] for s in spans).strip()))
                continue
            if inside(l["bbox"], skip_rects):
                continue
            raw.append(Line(x0, y0, x1, y1, spans))
    if footer_bits:
        # printed page number = right-most footer fragment that looks like a page label
        for _, t in sorted(footer_bits, reverse=True):
            m = re.search(r"(?:^|\s)(\d{1,3}|[ivxlc]+|[A-Z]-\d{1,3})$", t)
            if m:
                printed = m.group(1)
                break

    raw.sort(key=lambda ln: (ln.yc, ln.x0))
    merged: list[Line] = []
    for ln in raw:
        if merged and abs(merged[-1].yc - ln.yc) < SAME_LINE_DY:
            m = merged[-1]
            m.spans += ln.spans
            m.x0, m.y0, m.x1, m.y1 = min(m.x0, ln.x0), min(m.y0, ln.y0), max(m.x1, ln.x1), max(m.y1, ln.y1)
        else:
            merged.append(ln)
    return merged, printed


def classify(line: Line, bfont: str, bsize: float):
    """Return ('heading', level, text) | ('bullet', None, text) | ('text', None, text)."""
    spans = sorted(line.spans, key=lambda s: s[0])
    label = None
    if len(spans) > 1 and NUM_LABEL.match(spans[0][1].strip()):
        label = spans[0][1].strip().rstrip(".")
        rest = spans[1:]
    else:
        rest = spans
    rest_text = re.sub(r"\s+", " ", " ".join(s[1] for s in rest)).strip()
    heavy = all(HEAVY_FONT.search(s[2]) for s in rest if s[1].strip())
    max_size = max(s[3] for s in rest)
    short = len(rest_text) < 120 and not rest_text.endswith((".", ";", ","))

    if heavy and short and max_size >= bsize - 0.5 and rest_text:
        m = re.match(r"^(\d+(?:\.\d+)+)\s+(.*)$", rest_text)
        if not label and m:
            label, rest_text = m.group(1), m.group(2)
        if label:
            level = min(label.count(".") + 2, 6)
            return "heading", level, f"{label} {rest_text}"
        level = 2 if max_size >= bsize + 2.5 else 5
        return "heading", level, rest_text

    first = spans[0][1].strip()
    if first in BULLET_GLYPHS and len(spans) > 1:
        return "bullet", None, re.sub(r"\s+", " ", " ".join(s[1] for s in spans[1:])).strip()
    if re.match(r"^[•▪■➢◦]\s*", line.text):
        return "bullet", None, re.sub(r"^[•▪■➢◦]\s*", "", line.text)
    return "text", None, line.text


def join(a: str, b: str) -> str:
    if a.endswith("-") and not a.endswith(" -"):
        return a + b
    return a + " " + b


def render_page(page, bfont, bsize) -> tuple[str, str | None]:
    try:
        tabs = list(page.find_tables())
    except Exception:  # table finder can fail on odd pages; fall back to text only
        tabs = []
    tabs = [t for t in tabs if t.row_count >= 2 and t.col_count >= 2]
    rects = [pymupdf.Rect(t.bbox) for t in tabs]
    lines, printed = collect_lines(page, rects)

    # blocks: (y, kind, payload)
    blocks: list[tuple[float, str]] = [(t.bbox[1], table_markdown(t)) for t in tabs]

    para: list[str] = []
    para_y = None
    prev: Line | None = None
    prev_kind = None

    def flush():
        nonlocal para, para_y
        if para:
            blocks.append((para_y, para[0] if len(para) == 1 else para[0]))
        para, para_y = [], None

    # a line ending well short of the right margin ends its paragraph (lists, acronyms, etc.)
    right = max((ln.x1 for ln in lines), default=540)
    full_line_x1 = right - 95
    line_gap = bsize * 0.75  # gap (pt) between y1 of one line and y0 of next that still counts as same paragraph
    for ln in lines:
        kind, level, text = classify(ln, bfont, bsize)
        gap = ln.y0 - prev.y1 if prev else 999
        if kind == "heading":
            flush()
            # a wrapped heading: unnumbered heavy line right under a heading -> same heading
            if prev_kind == "heading" and gap <= line_gap and level == 5 and blocks:
                y, h = blocks[-1]
                blocks[-1] = (y, h + " " + text)
            else:
                blocks.append((ln.y0, "#" * level + " " + text))
            prev, prev_kind = ln, kind
            continue
        if kind == "bullet":
            flush()
            para, para_y = ["- " + text], ln.y0
        elif (para and gap <= line_gap and prev_kind in ("text", "bullet")
              and prev.x1 >= full_line_x1 and not TOC_LINE.search(prev.text)
              and not (LIST_START.match(text) and LIST_START.match(para[0]))):
            para[0] = join(para[0], text)
        else:
            flush()
            para, para_y = [text], ln.y0
        prev, prev_kind = ln, kind if kind != "bullet" else "bullet"
    flush()

    blocks.sort(key=lambda b: b[0])
    return "\n\n".join(b[1] for b in blocks if b[1]), printed


def parse_pages(spec: str, page_count: int) -> list[int]:
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        start, _, end = part.partition("-")
        s, e = int(start), int(end or start)
        if s < 1 or e > page_count or s > e:
            sys.exit(f"Bad page range {part!r} for a {page_count}-page PDF")
        pages.extend(range(s - 1, e))
    return pages


def extract(pdf_path: Path, out_path: Path, page_spec: str | None) -> None:
    doc = pymupdf.open(pdf_path)
    pages = parse_pages(page_spec, doc.page_count) if page_spec else list(range(doc.page_count))
    bfont, bsize = body_font(doc, pages)

    parts = [f"<!-- source: {pdf_path.name} | PDF pages {page_spec or f'1-{doc.page_count}'} of {doc.page_count} -->\n"]
    stats = []
    for i in pages:
        md, printed = render_page(doc[i], bfont, bsize)
        marker = f"<!-- page {i + 1}" + (f" | printed {printed}" if printed else "") + " -->"
        parts.append(f"\n{marker}\n\n{md}\n")
        stats.append({"page": i + 1, "printed": printed, "md_chars": len(md),
                      "plain_chars": len(doc[i].get_text().strip())})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")
    out_path.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")

    low = [s["page"] for s in stats if s["plain_chars"] < LOW_TEXT_THRESHOLD]
    print(f"{pdf_path.name}: {len(pages)} pages, body font {bfont} {bsize}pt, "
          f"{sum(s['md_chars'] for s in stats):,} chars -> {out_path}")
    if low:
        print(f"  low-text pages (<{LOW_TEXT_THRESHOLD} chars): {low}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--pages", help="1-based page ranges, e.g. 5-131")
    args = ap.parse_args()
    extract(args.pdf, args.out, args.pages)


if __name__ == "__main__":
    main()
