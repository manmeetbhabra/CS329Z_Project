"""Which model the agent uses, behind the project budget.

    from agent.config import configure
    configure("dev")        # gpt-6-luna via OpenRouter becomes DSPy's default LM; usage tracking on

Every LLM call goes through BudgetedLM, which asks utils.budget before the call is sent: it checks
OpenRouter's usage every few calls and, once PROJECT_BUDGET_USD (in .env) is spent, refuses every
call with BudgetExceeded. (A DSPy callback can't do this: DSPy logs and ignores callback errors.)
"""

from __future__ import annotations

import os
from pathlib import Path

import dspy
from dotenv import load_dotenv

from utils.budget import METER

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "dev": "openrouter/openai/gpt-6-luna",       # cheap: development and default runs ($0.10 / $0.50 per M tokens)
    "strong": "openrouter/openai/gpt-6-sol",     # ~20x the price: diagnostic runs ($2 / $10 per M tokens)
}


class BudgetedLM(dspy.LM):
    """A dspy.LM that checks the project budget before each call (cache hits included, cheaply:
    OpenRouter is only asked every few calls). It guards __call__/acall, the entry points DSPy's
    adapters use (`lm(messages=...)`), not forward(), whose overriding DSPy 3.4 deprecates."""

    def __call__(self, *args, **kwargs):
        METER.tick()
        return super().__call__(*args, **kwargs)

    async def acall(self, *args, **kwargs):
        METER.tick()
        return await super().acall(*args, **kwargs)


def make_lm(alias: str = "dev") -> dspy.LM:
    """make_lm("dev") -> BudgetedLM("openrouter/openai/gpt-6-luna", temperature=0.0, ...)"""
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is missing from .env")
    return BudgetedLM(MODELS[alias], temperature=0.0, max_tokens=4000, num_retries=3)


def configure(alias: str = "dev") -> dspy.LM:
    """Make the LM, set it as DSPy's default with the JSON adapter, and turn on token tracking.

    JSONAdapter: the model answers with one JSON object holding the signature's output fields.
    With gpt-6-luna, DSPy's default ChatAdapter sometimes left its own '[[ ## completed ## ]]'
    marker inside the last field; JSON gave clean typed fields in one call (checked 2026-11-08)."""
    lm = make_lm(alias)
    dspy.configure(lm=lm, adapter=dspy.JSONAdapter(), track_usage=True)
    return lm
