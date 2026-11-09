"""Remaining OpenRouter balance for the key in .env.

    uv run python -m utils.balance

Asks OpenRouter (GET /api/v1/key) for the key's spending limit and usage.
This is the provider's own count, so it includes every call made with this
key (DSPy runs, notebooks, anything else), not just one script's calls.

To see what a single run cost, take a reading before and after it:

    from utils.balance import spent_since, usage_now
    start = usage_now()
    ...  # run the agent / an eval
    print(f"run cost: ${spent_since(start):.4f}")

OpenRouter's usage figure can lag by a few seconds after the last call.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
KEY_URL = "https://openrouter.ai/api/v1/key"


def key_info() -> dict:
    """The "data" object OpenRouter returns for the key: limit, usage, limit_remaining, ..."""
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env or the environment.")
    request = urllib.request.Request(KEY_URL, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)["data"]


def usage_now() -> float:
    """Total dollars spent on this key so far, per OpenRouter."""
    return float(key_info().get("usage") or 0.0)


def spent_since(start_usage: float) -> float:
    """Dollars spent on this key since a usage_now() reading."""
    return usage_now() - start_usage


def balance_report() -> str:
    info = key_info()
    usage = info.get("usage") or 0.0
    limit = info.get("limit")
    remaining = info.get("limit_remaining")
    lines = [f"  used:      ${usage:.4f}"]
    for period in ("daily", "weekly", "monthly"):
        value = info.get(f"usage_{period}")
        if value is not None:
            lines.append(f"  {period + ':':<10} ${value:.4f}")
    if limit is None:
        lines.append("  limit:     none set on this key")
    else:
        lines.append(f"  limit:     ${limit:.2f}")
        lines.append(f"  remaining: ${remaining if remaining is not None else limit - usage:.4f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(balance_report())
