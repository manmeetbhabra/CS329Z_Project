# Mine Plan Review Agent

An LLM agent that reviews a draft mine permit application before it is submitted.

**Context.** To build a mine on U.S. federal land, a mining company has to submit a **Plan of Operations** to the
Bureau of Land Management (BLM). The plan is a document of several hundred pages that describes the mine, how
it will be monitored and how the land will be restored afterward. Federal rules (43 CFR 3809) list what it must
contain. Plans with missing or vague content get sent back, and each round trip can add months to permitting.

**What the agent does.** It reads a plan, checks it against a 51-item checklist derived from those rules, and flags
content that is missing, vague, inconsistent or implausible. Each flag cites the place in the plan where the problem is,
so a reviewer can fix it before the agency sees it.

**How it's evaluated.** The agent is measured on a test bench built from the Thacker Pass lithium mine plan. There are 41 copies of the plan:
**TP-000** is clean, and **TP-001 to TP-040** each have one planted flaw. A flag counts as a catch if it names
the right checklist item at the right place in the plan.

## Quickstart

You need Python 3.12+, [uv](https://docs.astral.sh/uv/) and an [OpenRouter](https://openrouter.ai) API key.

```bash
git clone <repo-url> && cd <repo>
uv sync                                  # installs everything into .venv
cp .env.example .env                     # then paste your OPENROUTER_API_KEY into .env
```

Check that everything works (free, no API calls):

```bash
uv run pytest -q
```

Run the agent:

```bash
uv run python -m harness.run v2 --cases smoke --name my-smoke --score   # a few cases
uv run python -m harness.run v2 --name my-run --score                   # all 41 cases (~4 min, ~$0.03)
```

Results go to `runs/<name>/`: one JSON report per case, plus `scores.md` with the summary.

**Notebook:** open [Demo.ipynb](Demo.ipynb) with the `.venv` kernel and run all cells. It does the same full run and prints the scores.

### Options

| Flag | Values |
|---|---|
| system (first argument) | `v2` = checklist + value checks (best), `v1` = checklist only, `replay` = no-AI pipeline check |
| `--model` | `dev` = gpt-6-luna (cheap, default), `strong` = gpt-6-sol (~20x the price) |
| `--cases` | `all` (default), `smoke`, `flawed`, or ids such as `TP-000,TP-021` |

**Spending limit (optional):** set `PROJECT_BUDGET_USD` in `.env`, and runs stop once that much has been spent on the key.
Answers that DSPy has already cached cost nothing.

## Repository layout

| Folder | What's in it |
|---|---|
| `agent/` | The reviewer: plan index, BM25 search, checklist, report writer, and the systems in `agent/systems/` |
| `harness/` | Runs a system over the bench, then scores and analyzes the results |
| `bench/` | Test cases, the hidden answer key and the scorer. See [bench/README.md](bench/README.md) for how cases are built |
| `knowledge/` | The 51-item checklist derived from 43 CFR 3809 |
| `corpus/` | The Thacker Pass plan, extracted to page-tagged Markdown |
| `prep/` | Scripts that turn the source PDFs into `corpus/` (you don't need these to run the agent) |
| `utils/` | OpenRouter usage and the budget guard |

The source PDFs aren't in the repo. You don't need them to run the agent, because `corpus/` holds their extracted text.
