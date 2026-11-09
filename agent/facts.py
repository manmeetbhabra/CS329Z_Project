"""The fact ledger: every quantity a plan states, grouped so copies of the same fact can be compared.
Plain code (no AI), so it is repeatable, testable and free; only reading chunks and judging the
groups that disagree need the model (agent/systems/v2.py).

    facts = [PlacedFact(Fact(subject="West WRSF", attribute="storage capacity", value="32.7", unit="M CY",
                             quote="approximately 32.7 M CY"), "body.md", "3.5", (51,)), ...]
    groups = group_facts(facts)            # copies of the same fact (same unit, similar attribute and subject)
    for g in disagreements(groups):        # groups whose copies disagree beyond rounding
        g.majority, g.odd                  # 41 (x4) vs 45 (x1): the 45 is the odd copy

Also: numeric_chunks(index) picks the chunks worth reading for numbers.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel

from agent.plan import PAGE_MARKER, Chunk, PlanIndex


class Fact(BaseModel):
    """One quantity stated as a fact, as the model reads it from a chunk."""
    subject: str            # "West WRSF"
    attribute: str          # "storage capacity"
    value: str              # "32.7" as written ("six to 12", "41", "2.5H:1V", "80 million")
    unit: str               # "M CY", "years", "lb PLS/acre", "" for a pure number
    quote: str              # the exact words that contain the value


@dataclass(frozen=True)
class PlacedFact:
    """A fact plus where it was read."""
    fact: Fact
    file: str
    section: str | None
    pages: tuple[int, ...]
    chunk_id: int = -1

    @property
    def place(self) -> tuple:
        return (self.file, self.section, self.pages)

    def as_dict(self) -> dict:
        return {**self.fact.model_dump(), "file": self.file, "section": self.section, "pages": list(self.pages),
                "chunk_id": self.chunk_id}


# ----------------------------------------------------------------------------------- values

_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
          "forty": 40, "fifty": 50, "hundred": 100, "half": 0.5, "one-half": 0.5, "quarter": 0.25}
_FRACTIONS = {"½": 0.5, "¼": 0.25, "¾": 0.75}
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")
_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+|[a-z]+(?:-[a-z]+)?")


def parse_value(text: str) -> tuple[float, float] | None:
    """A written value as (low, high); a single number has low == high. None if there is no number.
    "5,545.0" -> (5545.0, 5545.0);  "six to 12" -> (6.0, 12.0);  "18–24" -> (18.0, 24.0)
    "2.5H:1V" -> (2.5, 2.5);  "3:1" -> (3.0, 3.0);  "80 million" -> (80e6, 80e6);  "one-half" -> (0.5, 0.5)"""
    if not text:
        return None
    t = text.lower().replace("–", "-").replace("—", "-")
    for k, v in _FRACTIONS.items():
        t = t.replace(k, f" {v} ")
    ratio = re.search(r"(\d+(?:\.\d+)?)\s*h?\s*:\s*(\d+(?:\.\d+)?)\s*v?", t)
    if ratio:                                                     # slopes as H per 1 V
        h, v = float(ratio.group(1)), float(ratio.group(2))
        return (h / v, h / v) if v else None
    nums = []                                                     # digits and number words, in order
    for tok in _TOKEN.findall(t):
        if tok[0].isdigit() or tok[0] == ".":
            if tok.replace(",", "").replace(".", ""):
                nums.append(float(tok.replace(",", "")))
        elif tok in _WORDS:
            nums.append(float(_WORDS[tok]))
    if not nums:
        return None
    scale = next((s for w, s in _SCALE.items() if re.search(rf"\b{w}\b", t)), 1.0)
    nums = [n * scale for n in nums]
    lo, hi = (nums[0], nums[1]) if len(nums) >= 2 and re.search(r"\bto\b|-|through|and", t) else (nums[0], nums[0])
    return (min(lo, hi), max(lo, hi))


# ------------------------------------------------------------------------------------ units

_UNITS = [  # (canonical, factor, spellings) - longest spellings are tried first
    ("mcy", 1.0, ["million cubic yards", "million cy", "m cy", "mcy", "m cyd", "mm cy"]),
    ("cy", 1.0, ["cubic yards", "cubic yard", "cyds", "cyd", "cy"]),
    ("acre_ft", 1.0, ["acre-feet", "acre feet", "acre-foot", "af"]),
    ("acre", 1.0, ["acres", "acre", "ac"]),
    ("ft", 1.0, ["feet", "foot", "ft", "'"]),
    ("in", 1.0, ["inches", "inch", "in", '"']),
    ("mi", 1.0, ["miles", "mile", "mi"]),
    ("yr", 1.0, ["years", "year", "yrs", "yr"]),
    ("month", 1.0, ["months", "month"]),
    ("day", 1.0, ["days", "day"]),
    ("hr", 1.0, ["hours", "hour", "hrs", "hr"]),
    ("gpm", 1.0, ["gallons per minute", "gpm"]),
    ("gal", 1e6, ["million gallons"]),
    ("gal", 1.0, ["gallons", "gallon", "gal"]),
    ("tpy", 1.0, ["short tons per year", "tons per year", "tpy"]),
    ("ton", 1.0, ["short tons", "tons", "ton"]),
    ("lb_pls_acre", 1.0, ["pounds of pure live seed per acre", "lb pls/acre", "lb pls per acre", "pls lb/acre",
                          "lbs pls/acre", "lb/acre", "pounds per acre"]),
    ("pct", 1.0, ["percent", "%"]),
    ("deg", 1.0, ["degrees", "degree", "°"]),
    ("hv", 1.0, ["h:v", "h:1v", "horizontal to vertical", "h"]),
    ("hp", 1.0, ["horsepower", "hp"]),
    ("mph", 1.0, ["miles per hour", "mph"]),
]
_SPELLINGS = sorted(((s, c, f) for c, f, ss in _UNITS for s in ss), key=lambda x: -len(x[0]))


def normalize_unit(unit: str) -> tuple[str, float]:
    """A written unit as (canonical, factor to apply to the value).
    "M CY" -> ("mcy", 1.0);  "million cubic yards" -> ("mcy", 1.0);  "yrs" -> ("yr", 1.0)
    "million gallons" -> ("gal", 1e6);  "" -> ("", 1.0);  "lb PLS/acre" -> ("lb_pls_acre", 1.0)
    Anything else -> its lowercase words joined by "_" ("tons of ore per day" -> "tons_of_ore_per_day")."""
    u = " ".join(unit.lower().replace("°", "°").split())
    if not u:
        return "", 1.0
    for spelling, canonical, factor in _SPELLINGS:
        if u == spelling:
            return canonical, factor
    return "_".join(re.findall(r"[a-z0-9%°]+", u)), 1.0


def canonical_value(f: Fact) -> tuple[float, float] | None:
    """parse_value(f.value) scaled by the unit's factor: (80, "million gallons") -> (8e7, 8e7)."""
    v = parse_value(f.value)
    if v is None:
        return None
    _, factor = normalize_unit(f.unit)
    return (v[0] * factor, v[1] * factor)


# ---------------------------------------------------------------------------------- grouping

_STOP = {"the", "and", "of", "for", "per", "approximately", "approx", "about", "each", "will", "be", "a", "an",
         "to", "in", "on", "at", "by", "is", "are", "with", "from", "proposed", "estimated", "project's"}
GENERIC_SUBJECTS = {"project", "plan", "mine", "site", "operation", "operations", "lnc", "facility", "proposed action"}
ATTRIBUTE_JACCARD = 0.5
SUBJECT_JACCARD = 0.5


def _words(text: str) -> frozenset[str]:
    """Content words, lightly stemmed: "Storage Capacities" -> {"storage", "capacitie"} -> plural 's' dropped."""
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return frozenset(out)


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def same_subject(a: str, b: str) -> bool:
    """"West WRSF" ~ "the West WRSF facility"; "Project" ~ "the Project"; "Project" !~ "CTFS"."""
    wa, wb = _words(a), _words(b)
    ga, gb = (" ".join(sorted(wa)) in GENERIC_SUBJECTS or not wa), (" ".join(sorted(wb)) in GENERIC_SUBJECTS or not wb)
    if ga or gb:
        return ga and gb
    return wa <= wb or wb <= wa or _jaccard(wa, wb) >= SUBJECT_JACCARD


def _generic_or_empty(a: str, b: str) -> bool:
    """True if either subject is generic ("Project", "the mine"): with an identical attribute it may
    be the same fact as a specific subject ("Project mine life" vs "open pit mine life")."""
    return any(" ".join(sorted(_words(s))) in GENERIC_SUBJECTS or not _words(s) for s in (a, b))


def same_attribute(a: str, b: str) -> bool:
    """"storage capacity" ~ "capacity (storage)"; "mine life" ~ "life of mine"; "slope" !~ "height"."""
    wa, wb = _words(a), _words(b)
    return wa == wb or _jaccard(wa, wb) >= ATTRIBUTE_JACCARD


def group_facts(facts: list[PlacedFact]) -> list[list[PlacedFact]]:
    """Copies of the same fact: same canonical unit, same attribute (Jaccard >= 0.5 on content words)
    and compatible subjects. Single-link groups (union-find) within each unit.
    [mine life 41 yr (Summary), mine life 41 yr (§3.3.2), mine life 45 yr (§3.4), phase 1 duration 4 yr]
    -> [[41, 41, 45], [4]]"""
    by_unit: dict[str, list[int]] = {}
    for i, pf in enumerate(facts):
        if canonical_value(pf.fact) is not None:
            by_unit.setdefault(normalize_unit(pf.fact.unit)[0], []).append(i)
    parent = list(range(len(facts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for idx in by_unit.values():
        for x in range(len(idx)):
            fa = facts[idx[x]].fact
            for y in range(x + 1, len(idx)):
                fb = facts[idx[y]].fact
                if same_attribute(fa.attribute, fb.attribute) and (
                        same_subject(fa.subject, fb.subject)
                        or (_words(fa.attribute) == _words(fb.attribute) and _generic_or_empty(fa.subject, fb.subject))):
                    parent[find(idx[x])] = find(idx[y])
    groups: dict[int, list[PlacedFact]] = {}
    for idx in by_unit.values():
        for i in idx:
            groups.setdefault(find(i), []).append(facts[i])
    return list(groups.values())


@dataclass
class Group:
    """A group of copies whose values disagree."""
    facts: list[PlacedFact]
    majority: tuple[float, float] | None            # the most common value (None if every value differs)
    odd: list[int] = field(default_factory=list)     # indices into facts of copies that differ from the majority

    @property
    def key(self) -> str:
        f = self.facts[0].fact
        return f"{f.subject} | {f.attribute} | {normalize_unit(f.unit)[0]}"


def _close(a: tuple[float, float], b: tuple[float, float], tol: float = 0.01) -> bool:
    return all(abs(x - y) <= tol * max(abs(x), abs(y), 1e-9) for x, y in zip(a, b))


def disagreements(groups: list[list[PlacedFact]], max_facts: int = 12) -> list[Group]:
    """Groups seen in >= 2 places whose values differ by more than rounding (1%), with the odd copies.
    Exact duplicates of one fact at one place are collapsed first. Groups larger than max_facts keep
    the odd copies and as many majority copies as fit.
    [41, 41, 41, 45] -> Group(majority=(41, 41), odd=[3]);  [3, 5] -> Group(majority=None, odd=[0, 1])"""
    out = []
    for g in groups:
        uniq, seen = [], set()
        for pf in g:
            key = (pf.place, canonical_value(pf.fact))
            if key not in seen:
                seen.add(key)
                uniq.append(pf)
        if len({pf.place for pf in uniq}) < 2:
            continue
        values = [canonical_value(pf.fact) for pf in uniq]
        clusters: list[tuple[tuple[float, float], int]] = []
        for v in values:
            for i, (c, n) in enumerate(clusters):
                if _close(v, c):
                    clusters[i] = (c, n + 1)
                    break
            else:
                clusters.append((v, 1))
        if len(clusters) < 2:
            continue
        clusters.sort(key=lambda cn: -cn[1])
        majority = clusters[0][0] if clusters[0][1] > clusters[1][1] else None
        odd = [i for i, v in enumerate(values) if majority is None or not _close(v, majority)]
        if len(uniq) > max_facts:
            keep = odd + [i for i in range(len(uniq)) if i not in odd][: max(0, max_facts - len(odd))]
            uniq = [uniq[i] for i in keep]
            odd = list(range(len([i for i in keep if i in odd])))
        out.append(Group(uniq, majority, odd))
    return out


# ------------------------------------------------------------------------ table arithmetic

def _cell_number(cell: str) -> float | None:
    """A table cell as a number. Footnote marks are dropped; a value in parentheses is one the
    table's total excludes (it's counted elsewhere), so it counts as 0; a dash is 0.
    "1,166.1" -> 1166.1;  "50.5 [2]" -> 50.5;  "(50.5) [2]" -> 0.0;  "-" -> 0.0;  "3:1", "n/a", "" -> None"""
    c = re.sub(r"\s*\[\d+\]", "", cell.strip()).replace(",", "").strip()
    if c in ("-", "–", "—", "--"):
        return 0.0
    if re.fullmatch(r"\(\s*-?\d+(?:\.\d+)?\s*\)", c):
        return 0.0
    return float(c) if re.fullmatch(r"-?\d+(?:\.\d+)?", c) else None


_TOTAL_ROW = re.compile(r"^\s*\**\s*(grand\s+)?total\b", re.I)


def table_total_checks(text: str, tol: float = 0.005) -> list[str]:
    """Markdown tables whose 'Total' row disagrees with the sum of the rows above it (since the
    header or the previous total), column by column. Plain arithmetic, as notes for the reader.
    -> ['Table row "Total": column "Acres" rows sum to 5,995.0 but the total says 5,545.0']"""
    notes = []
    rows: list[list[str]] = []
    header: list[str] = []
    for line in text.splitlines() + [""]:
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            rows, header = [], []
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue                                                    # the |---|---| line
        if not header:
            header = cells
            continue
        if cells and _TOTAL_ROW.match(cells[0]):
            for col in range(1, len(cells)):
                total = _cell_number(cells[col])
                parts = [_cell_number(r[col]) if col < len(r) else None for r in rows]
                if total is None or len(parts) < 2 or any(p is None for p in parts):
                    continue                                            # only columns that are fully numeric
                if sum(1 for p in parts if p) < 2:
                    continue                                            # mostly dashes/zeros: nothing to add up
                s_ = sum(parts)
                if abs(s_ - total) > tol * max(abs(total), 1.0):
                    name = header[col] if col < len(header) else f"column {col + 1}"
                    notes.append(f'Table row "{cells[0]}": column "{name}" rows sum to {s_:,.1f} '
                                 f"but the total says {total:,.1f}")
            rows = []
        elif any(c for c in cells[1:]):                                 # skip label rows ("Proposed Project | | |")
            rows.append(cells)
    return notes


# ---------------------------------------------------------------------------- which chunks

_UNIT_WORDS = (r"(?:%|°|percent|acres?|feet|foot|ft|inch(?:es)?|miles?|years?|yrs?|months?|days?|hours?"
               r"|gallons?|gpm|tons?|tpy|cubic|cy\b|m cy|million|pounds?|lbs?|degrees?)")
_NUMBER_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|fifty|hundred|half)"
_QUANTITY = re.compile(
    rf"\d[\d,]*(?:\.\d+)?\s*(?:{_UNIT_WORDS}|h\s*:|:\s*1)"                       # 32.7 M CY, 3H:1V
    rf"|\b{_NUMBER_WORDS}(?:-{_NUMBER_WORDS})?\s+{_UNIT_WORDS}"                   # six inches, three years
    r"|\b(?:year|years)\s+\d{1,2}\b|\b\d+(?:\.\d+)?\s*h\s*:\s*\d+\s*v\b"           # Year 41, 2.5H:1V
    r"|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b"
    r"|\b\d[\d,]*\s+(?:[a-z-]+\s+){0,2}[a-z]+s\b", re.I)                            # 429 unpatented lode claims
_TABLE_NUMBERS = re.compile(r"^\|.*\d.*\|.*\d.*\|", re.M)
_SKIP_HEADINGS = ("acronym", "abbreviat", "list of", "units of measure")


def numeric_chunks(index: PlanIndex) -> list[Chunk]:
    """Chunks worth reading for numbers: a quantity with a unit (in digits or words: "32.7 M CY",
    "six inches"), a ratio, a "Year N", a calendar date, a count of things ("429 unpatented lode
    claims"), or a table row with numbers; acronym and abbreviation lists are skipped. 335 of a
    Thacker Pass case's 515 chunks. (A narrower first version read 153 and missed four planted values.)"""
    out = []
    for c in index.chunks:
        if any(s in c.path.lower() for s in _SKIP_HEADINGS):
            continue
        text = PAGE_MARKER.sub(" ", index.chunk_text(c))
        if _QUANTITY.search(text) or _TABLE_NUMBERS.search(text):
            out.append(c)
    return out
