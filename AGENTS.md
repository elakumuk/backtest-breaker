# AGENTS.md — Backtest Breaker (Meta Global AI Developer Hackathon)

Read this before every task. These rules come from the hackathon Official Rules, and breaking one can disqualify the project.

## Project
Backtest Breaker: an AI system that tries to break a trading-strategy backtest before the market does. It **audits** strategies and never gives investment advice.
Full plan: see `PLAN.md` (day-by-day). Do only the task you are given, nothing beyond it.

## Hard rules
1. **New work only.** Everything must be written now (Oct 26 – Nov 4, 2026). Do not copy code from any other repo on this machine (strategy-audit-lab, SSGA-Summer2026, etc.).
2. **Public repo.** Never commit `.env`, API keys, or anything under `data/`. Check `git status` before every commit.
3. **Data rights.** Download Ken French / FRED data at runtime only. Never commit Yahoo/yfinance data.
4. **Runtime AI = Meta models only.** All LLM calls inside the product go through `bb/llm.py` and use a Meta model (Muse Spark via the Meta Model API, or Llama open weights via Baseten/Fireworks). Never call OpenAI/Anthropic models from product code.
5. **Numbers come from code, not the LLM.** Sharpe, DSR, PBO and holdout results are computed in Python. The LLM proposes, chooses and explains.
6. **No look-ahead.** Weights at date t use data up to t-1 only. Every engine change must keep `tests/test_engine.py::test_no_lookahead` green.

## Working style
- Python 3.11, typed, docstrings, small functions. pandas/numpy, pydantic, pytest.
- After each task: run `pytest`, then append one line to `BUILD_LOG.md`:
  `YYYY-MM-DD | tool=Codex | <what was built> | <what broke / was fixed>`
- Commit message format: `D<n>: <summary>`.
- If anything in the task conflicts with these rules, stop and say so instead of guessing.
