"""The 51 requirements from knowledge/checklist.yaml, in one shape every part can use.

    items = load_checklist()          # list[Item], in file order
    items_by_id()["rec-riparian"]     # one Item
    item.render()                     # the item as text for a prompt
    item.query()                      # the item as a plain search query (what V0's search check uses)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = ROOT / "knowledge" / "checklist.yaml"


@dataclass(frozen=True)
class Item:
    id: str
    cite: str
    section: str              # which part of the plan it belongs to: operator, operations, reclamation, ...
    text: str                 # the requirement in plain language
    kind: str                 # always | conditional | triggered
    applies_when: str
    vague_if: str
    catches: tuple[str, ...]
    trigger_facts: tuple[str, ...] = field(default=())

    def render(self) -> str:
        lines = [f"{self.id} ({self.cite}; {self.kind})", f"Requirement: {self.text}",
                 f"Applies when: {self.applies_when}", f"Too vague if: {self.vague_if}"]
        return "\n".join(lines)

    def query(self) -> str:
        """A plain search query: the requirement text, plus when it applies if that adds words."""
        extra = "" if self.applies_when.strip().lower().startswith("always") else " " + self.applies_when
        return self.text + extra


@lru_cache(maxsize=4)
def load_checklist(path: Path = CHECKLIST) -> tuple[Item, ...]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return tuple(Item(id=d["id"], cite=d["cite"], section=d["section"], text=" ".join(d["text"].split()),
                      kind=d["kind"], applies_when=" ".join(str(d["applies_when"]).split()),
                      vague_if=" ".join(str(d["vague_if"]).split()), catches=tuple(d.get("catches") or ()),
                      trigger_facts=tuple(d.get("trigger_facts") or ()))
                 for d in data["items"])


def items_by_id(path: Path = CHECKLIST) -> dict[str, Item]:
    return {i.id: i for i in load_checklist(path)}


def render_catalog(path: Path = CHECKLIST, max_chars: int = 110) -> str:
    """All items, one short line each, for a model to tag a finding with a checklist id:
    "op-identity: Name, mailing address and phone number of the operator(s).\\nop-tin: …" """
    return "\n".join(f"{i.id}: {i.text if len(i.text) <= max_chars else i.text[:max_chars].rstrip() + '…'}"
                     for i in load_checklist(path))
