"""Exact find/replace edits, shared by the gold-set build and the flaw-case build.

An edit is a mapping:
    file:    target file name (e.g. body.md)       -- used by callers to pick the text
    find:    exact text that must occur exactly once in the target
    replace: replacement text ("" deletes)
    note:    optional free-text reason (ignored here)

bench.build_cases also accepts {file: X, delete_file: true}, which leaves document X out of a
test case entirely (handled there, not here).

Edits are applied in order. Any edit whose `find` text matches zero times or more than once
raises PatchError, so a stale or ambiguous anchor fails loudly instead of silently.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


class PatchError(ValueError):
    pass


def apply_edit(text: str, edit: Mapping, where: str = "") -> str:
    find = edit["find"]
    replace = edit.get("replace", "")
    if not find:
        raise PatchError(f"{where}: empty `find`")
    n = text.count(find)
    if n != 1:
        preview = find if len(find) <= 80 else find[:77] + "..."
        raise PatchError(f"{where}: `find` matched {n} times (must be exactly 1): {preview!r}")
    return text.replace(find, replace, 1)


def apply_edits(files: dict[str, str], edits: Iterable[Mapping], where: str = "") -> dict[str, str]:
    """Apply edits to a {file name: text} mapping and return a new mapping."""
    out = dict(files)
    for i, edit in enumerate(edits):
        name = edit["file"]
        if name not in out:
            raise PatchError(f"{where} edit #{i + 1}: unknown file {name!r}")
        out[name] = apply_edit(out[name], edit, where=f"{where} edit #{i + 1} ({name})")
    return out
