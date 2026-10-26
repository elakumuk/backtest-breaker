# Backtest Breaker — Build Plan (Meta Global AI Developer Hackathon)

> **One line:** An AI that tries to break your backtest before the market does.
> **Track:** Indie Developer · **Theme:** Build for Your World (quant finance)
> **Deadline:** Wed Nov 4, 9:00 PM ET — public repo + 3-min demo video

---

## 0. How to use this file (Ela)

- Each day has a **"Prompt for Muse Code"** block. Paste it as-is into Muse Code.
- Don't move on until that day's **Acceptance** list passes. If something fails, paste the error back to Muse Code.
- Commit at the end of every day: `git commit -m "D<n>: ..."`. That history is how the judges see that everything was built during the event.
- Every day, add one line to `BUILD_LOG.md` covering what Muse Code wrote, what you changed, and what broke. That log feeds the "Use of Meta AI Tooling" (25%) score and the demo story.

---

## 1. Rules → design constraints (do not break)

| Rule | What it means for us |
|---|---|
| New work only (Oct 26–Nov 4) | New repo, first commit today. No code copied from strategy-audit-lab or SSGA. The old 1.17 → −0.37 result is **story only**. |
| Judges must see what's new | `CONTRIBUTIONS.md` + daily commits + `BUILD_LOG.md`. |
| Public repo | `.env` and `data/` are in `.gitignore`. No API keys, no downloaded data in git. |
| Data rights | Use **Ken French Data Library** (research use, downloaded at runtime) and **FRED**. Never commit Yahoo data. |
| Not "AI financial advice" | The product audits strategies. It never recommends trades. Say this in the README and the demo. |
| Materials are non-confidential | Nothing private (other ideas, SSGA work) goes into the repo, Discord or the demo. |

---

## 2. Architecture

```
                 ┌────────────────────────────┐
                 │  Research Agent (Muse Spark)│  proposes strategy specs,
                 │  "the overfitter"           │  iterates on IN-SAMPLE results
                 └─────────────┬──────────────┘
                               │ JSON StrategySpec
                               ▼
┌──────────────┐   ┌──────────────────────────┐   ┌───────────────────────┐
│ Data Loader  │──▶│ Deterministic Engine      │──▶│ Audit Battery (code)   │
│ Ken French   │   │ signal(t) uses data ≤ t-1 │   │ walk-forward, DSR,     │
│ runtime dl   │   │ costs, turnover, returns  │   │ param sensitivity,     │
└──────────────┘   └──────────────────────────┘   │ cost stress, PBO,      │
                                                   │ SEALED HOLDOUT         │
                                                   └──────────┬────────────┘
                                                              ▼
                                        ┌─────────────────────────────────┐
                                        │ Auditor Agent (Muse Spark)       │
                                        │ tool-calls extra attacks, writes │
                                        │ verdict: SURVIVES / FRAGILE /    │
                                        │ BROKEN + why                     │
                                        └─────────────────────────────────┘

Notebook Mode:  user .ipynb ──▶ Muse Code/Spark reads it ──▶ finds look-ahead /
                leakage ──▶ WRITES a pytest that fails on the bug ──▶ we run it ──▶ proof
```

**Why this split (the WHY):**
- **The engine and the statistics are plain code, not the LLM.** Numbers have to be reproducible, and an LLM that "computes" a Sharpe ratio can't be audited.
- **The LLMs only do judgement work:** proposing strategies, choosing attacks, explaining results, reading code. That's where they add value, and it's what the Meta-tooling score rewards.
- **The research agent is deliberately an overfitter.** Left to iterate in-sample, it reproduces the exact failure we're auditing for. The demo shows our own AI fooling itself, then the auditor catching it.
- **The sealed holdout is the core idea.** The holdout data is hashed and locked. The agent has to log a written prediction before it can be opened, and it can be opened only once. That makes "peeking" impossible and visible.

---

## 3. Repo layout

```
backtest-breaker/
  README.md            CONTRIBUTIONS.md   BUILD_LOG.md   LICENSE (MIT)
  .gitignore           (.env, data/, __pycache__, .ipynb_checkpoints)
  .env.example         META_API_KEY=
  pyproject.toml / requirements.txt
  bb/
    data.py            # Ken French + FRED runtime download, local cache in data/
    spec.py            # pydantic StrategySpec
    engine.py          # deterministic backtest
    metrics.py         # Sharpe, maxDD, turnover, Deflated Sharpe, PBO
    audit.py           # audit battery + sealed holdout vault
    llm.py             # Meta Model API client (one place)
    agents/research.py # overfitter loop
    agents/auditor.py  # tool-calling auditor
    notebook_mode.py   # ipynb → findings → generated pytest → run
  app/streamlit_app.py # demo UI
  tests/               # our own tests (lookahead, determinism, metrics)
  examples/leaky_notebook.ipynb   # a notebook WE write with planted bugs
```

---

## 4. Day plan (bug-free demo > feature count)

| Day | Date | Goal |
|---|---|---|
| D1 | Mon 10/26 | Repo, API hello-world, data loader |
| D2 | Tue 10/27 | Spec + engine + look-ahead tests |
| D3 | Wed 10/28 | Metrics + audit battery + sealed holdout |
| D4 | Thu 10/29 | Research agent (overfitter loop) |
| D5 | Fri 10/30 | Notebook Mode (moved up for the Meta-tooling score) |
| D6 | Sat 10/31 | Auditor agent (tool calling) |
| D7 | Sun 11/1 | Streamlit UI, end-to-end run |
| D8 | Mon 11/2 | Hardening, README, CONTRIBUTIONS, rehearse demo |
| D9 | Tue 11/3 | Record 3-min demo video |
| D10 | Wed 11/4 | Buffer, final checks, **submit by 9 PM ET (aim for noon)** |

---

### D1 — Repo + API + data

**Ela does by hand first:**
1. Create a new **public** GitHub repo called `backtest-breaker`, with no template.
2. Clone it locally. Put `META_API_KEY=...` in `.env`. Never commit that file.

**Prompt for Muse Code:**
```
We are building "Backtest Breaker", a Python 3.11 project: an AI system that audits
trading-strategy backtests for overfitting and look-ahead bias. It never gives
trading advice.

Today, set up the skeleton only:
1. Create the repo layout:
   bb/{__init__,data,spec,engine,metrics,audit,llm,notebook_mode}.py,
   bb/agents/{__init__,research,auditor}.py, app/, tests/, examples/.
   Add README.md, CONTRIBUTIONS.md, BUILD_LOG.md, MIT LICENSE, requirements.txt.
2. .gitignore must include: .env, data/, __pycache__/, .ipynb_checkpoints/, *.pyc.
   Add .env.example with META_API_KEY=.
3. bb/llm.py: a single small client for the Meta Model API that reads META_API_KEY
   from .env (python-dotenv). Function: chat(messages, model, tools=None,
   response_format=None) -> response. Look up the current Meta Model API docs for the
   exact endpoint and model names (Muse Spark). Add a retry with backoff and log
   token usage to stdout.
4. bb/data.py: download at RUNTIME (never commit data) from the Ken French Data
   Library:
     - "49_Industry_Portfolios_daily" (value-weighted daily returns)
     - "F-F_Research_Data_Factors_daily" (for RF)
   Parse the zipped CSVs into pandas DataFrames with a DatetimeIndex, returns as
   decimals (divide by 100), missing values (-99.99 / -999) set to NaN.
   Cache to data/ as parquet; re-use the cache if present.
   Function: load_universe() -> (returns_df, rf_series).
5. A script scripts/hello.py that (a) calls Muse Spark with "Say hello in 5 words",
   (b) loads the universe and prints its shape and date range.
Write clean, typed, documented code. Do not add features beyond this list.
```

**Acceptance:**
- [ ] `python scripts/hello.py` prints a Muse Spark reply and the data shape (~49 columns, daily since 1926).
- [ ] `git status` doesn't show `.env` or `data/`.
- [ ] First commit pushed: `D1: skeleton, Meta API client, Ken French loader`.

---

### D2 — Spec + engine + look-ahead tests

**Prompt for Muse Code:**
```
Add the strategy spec and a deterministic backtest engine.

1. bb/spec.py: pydantic model StrategySpec with fields:
   name, signal_type in {"momentum","reversal","low_vol","trend"},
   lookback_days (int 5..504), skip_days (int 0..42), top_n (1..24),
   rebalance in {"D","W","M"}, long_short (bool), cost_bps (float, default 10),
   start, end (dates). Include .to_json() / .from_json().
2. bb/engine.py: run_backtest(spec, returns, rf) -> BacktestResult
   (daily portfolio returns, weights, turnover).
   HARD RULE: weights decided at date t may only use returns up to t-1
   (shift by one). Equal-weight the top_n (and bottom_n if long_short).
   Apply transaction costs = cost_bps * turnover. Fully vectorized, deterministic,
   no randomness.
3. tests/test_engine.py:
   - test_no_lookahead: build a synthetic returns frame, then change ONLY the
     returns on the last day; assert that the weights on every day up to and
     including the last day are unchanged.
   - test_deterministic: the same spec run twice gives identical results.
   - test_costs: cost_bps=0 vs 50 differ by exactly cost * turnover.
Run pytest and fix until it is green.
```

**Acceptance:** `pytest` is green, and a momentum spec run on 1990–2015 gives a plausible Sharpe (roughly 0–1.5, not 5).

---

### D3 — Metrics + audit battery + sealed holdout

**Prompt for Muse Code:**
```
Implement metrics and the audit battery. All numbers are computed by code, not the LLM.

bb/metrics.py:
- sharpe (annualized, excess of rf), max_drawdown, annual_turnover.
- deflated_sharpe_ratio(sr, n_trials, sr_variance_across_trials, n_obs, skew, kurt)
  following Bailey & Lopez de Prado (2014). Cite the paper in the docstring.
- pbo_cscv(matrix_of_trial_returns, n_splits=16): Probability of Backtest
  Overfitting via Combinatorially Symmetric Cross-Validation
  (Bailey, Borwein, Lopez de Prado, Zhu 2016).

bb/audit.py:
- walk_forward(spec, ...): rolling 5y train / 1y test; report the out-of-sample
  Sharpe per fold.
- param_sensitivity(spec): perturb lookback/top_n by +/-20%; report the Sharpe
  spread. A big spread means FRAGILE.
- cost_stress(spec): Sharpe at 5/10/25/50 bps.
- class HoldoutVault: the last 3 years of data are SEALED.
    * seal(): stores the SHA256 hash of the holdout slice and a log file.
    * register_prediction(spec, predicted_sharpe, rationale) MUST be called first.
    * open_once(spec): runs the spec on the holdout only if a prediction is logged
      and the vault has never been opened; then it is permanently locked
      (state in an audit_log.jsonl).
    * Any attempt to read the holdout early raises HoldoutSealedError.
- run_audit(spec, trials_history) -> AuditReport (dataclass, JSON-serializable).

tests/test_audit.py: the vault refuses to open twice and refuses to open without a
prediction; DSR falls when n_trials rises; walk-forward folds don't overlap.
```

**Acceptance:** Tests are green, and `run_audit` returns readable JSON.

---

### D4 — Research Agent ("the overfitter")

**Prompt for Muse Code:**
```
bb/agents/research.py: an agent loop using Muse Spark via bb/llm.py.

- The system prompt says it is a quant researcher trying to MAXIMIZE in-sample Sharpe
  on 1990-(holdout start). It must answer ONLY with a JSON StrategySpec
  (use structured output / JSON schema from StrategySpec).
- The loop runs N=25 iterations: propose a spec -> run_backtest in-sample -> feed
  back the Sharpe and turnover -> it proposes a better spec.
- Every trial is logged to trials.jsonl (spec, in-sample Sharpe, timestamp). This
  log is what the Deflated Sharpe and PBO use as n_trials. NEVER hide trials.
- Invalid JSON: retry up to 2 times, then skip.
- At the end: pick the best in-sample spec, and have the agent write a predicted
  holdout Sharpe + rationale -> HoldoutVault.register_prediction.
CLI: python -m bb.agents.research --iters 25
```

**Acceptance:** 25 trials logged, the best in-sample Sharpe is visibly higher than the first trial's, and a prediction is registered.

**Demo note:** This is the "watch AI fool itself" moment. Save the output.

---

### D5 — Notebook Mode (biggest Meta-tooling moment)

**Ela does first:** Ask Muse Code to write `examples/leaky_notebook.ipynb` with **3 planted bugs**, then check by hand that the bugs are really there:
1. A signal that uses same-day returns (missing `shift(1)`).
2. Z-score normalization over the full sample (future leaks into the past).
3. The universe chosen with today's information (survivorship-style filter on the full period).

**Prompt for Muse Code:**
```
bb/notebook_mode.py: audit a user's Jupyter notebook for backtest bugs.

1. Parse the .ipynb (nbformat) and extract the code cells in order.
2. Send them to Muse Spark with a system prompt listing bug classes: look-ahead
   (missing shift, same-bar execution), full-sample normalization/fit leakage,
   survivorship/universe selection with future info, ignoring costs, multiple
   testing without correction. Ask for JSON findings:
   [{cell_index, line, bug_class, explanation, severity}].
3. For EACH finding, ask the model to WRITE a pytest test file
   (tests_generated/test_finding_<k>.py) that imports the notebook's logic
   (convert the notebook to a module with nbconvert) and FAILS because of the bug,
   e.g. perturb future data and show that past signals change.
4. Run the generated tests in a subprocess with a timeout. A finding is
   "PROVEN" only if its test fails for the stated reason; otherwise "UNPROVEN".
5. Return a NotebookReport. Never execute user code outside the subprocess with a
   timeout.
CLI: python -m bb.notebook_mode examples/leaky_notebook.ipynb
```

**Acceptance:** At least 2 of the 3 planted bugs are marked **PROVEN**, and there are 0 false "PROVEN" results on a clean twin notebook (write a corrected copy and run that too).

---

### D6 — Auditor Agent (tool calling)

**Prompt for Muse Code:**
```
bb/agents/auditor.py: Muse Spark with TOOL CALLING over our audit functions.
Tools: walk_forward, param_sensitivity, cost_stress, deflated_sharpe (with the real
n_trials from trials.jsonl), pbo_cscv, open_holdout_once.
- Goal in the system prompt: "Try to break this strategy. Choose which attacks to
  run and in what order. You may open the holdout exactly once, last."
- It must cite the actual numbers returned by the tools; it may not invent numbers.
  After the run, verify: every number in the verdict text must appear in the tool
  outputs (simple regex check); flag any that don't.
- Output: verdict in {SURVIVES, FRAGILE, BROKEN}, a 5-bullet explanation in plain
  English for a non-quant, and the full tool-call trace.
CLI: python -m bb.agents.auditor --spec best_spec.json
```

**Acceptance:** On the overfitter's best spec, the auditor lands on FRAGILE/BROKEN with real numbers. On a simple 12-1 momentum spec, it gives a more lenient verdict. If it rules both BROKEN, the auditor is too aggressive, and that needs a fix.

---

### D7 — Streamlit UI + end-to-end

**Prompt for Muse Code:**
```
app/streamlit_app.py with 3 tabs:
1. "Watch it overfit": run/replay the research agent; a live chart of in-sample
   Sharpe per trial.
2. "Break it": run the auditor on the chosen spec; show the tool-call trace as a
   timeline, then the verdict card, then in-sample vs holdout equity curves, with
   the predicted vs actual holdout Sharpe side by side.
3. "Audit my notebook": upload an .ipynb, then show the findings with
   PROVEN/UNPROVEN badges and the generated test code.
A footer on every page: "Audits strategies; does not give investment advice."
Cache heavy calls; add a "demo mode" that replays saved runs so the video never
depends on live API latency.
```

**Acceptance:** A full run with `streamlit run app/streamlit_app.py`, plus the same thing in demo mode with no internet.

---

### D8 — Hardening + docs

- README: problem → demo GIF → architecture diagram → how to run → **"What was built during the hackathon"** → limitations.
- CONTRIBUTIONS.md: list every module with the date it was built (from the commit history).
- BUILD_LOG.md: clean it up, and make it say clearly how Muse Code / Muse Spark were used.
- A fresh-clone test: `git clone` into another folder → `pip install` → do the README steps exactly as written. Anything that breaks, fix it.
- Rehearse the demo twice.

---

### D9 — Demo video (3:00 max)

| Time | Scene |
|---|---|
| 0:00–0:20 | Hook: "In a past project, a strategy with a 1.17 Sharpe went to −0.37 the moment it met unseen data. Most backtests lie, and nobody checks them." |
| 0:20–0:50 | Tab 1: Muse Spark overfits live, and the Sharpe climbs. "Our own AI just fooled itself." |
| 0:50–1:40 | Tab 2: the auditor picks its attacks (tool trace), DSR with real n_trials, the sealed holdout opens ONCE, prediction vs reality. Verdict: BROKEN. |
| 1:40–2:30 | Tab 3: upload a notebook → Muse finds look-ahead → **writes a test that proves it** → the test fails red. "It doesn't just say there's a bug. It proves it." |
| 2:30–3:00 | Who it's for (students, retail quants, small funds), what's next, and "built in 10 days on Muse Spark + Muse Code." |

Tips: use demo mode, record at 1080p, keep the screen free of personal info, and put your voice on top. Subtitles help.

---

### D10 — Submit (by noon, not 9 PM)

- [ ] The repo is **public** and the README is complete
- [ ] The repo has no `.env` and no data (`git log --all -- .env` comes back empty)
- [ ] The demo video link opens (YouTube unlisted / Loom)
- [ ] Indie Developer track selected
- [ ] Submit: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit
- [ ] Post on LinkedIn/X with the demo link (there's a bonus prize for most viral / community vote)

---

## 5. Judging map (where each point comes from)

| Criterion | Weight | Our evidence |
|---|---|---|
| Impact | 30% | Real problem (overfit backtests); clear users; not advice |
| Meta tooling | 25% | Muse Spark structured output (research), tool calling (auditor), **code reading + test generation (notebook mode)**, Muse Code built the repo (BUILD_LOG) |
| Technical depth | 20% | Deterministic engine, look-ahead tests, DSR, PBO, sealed vault, number-grounding check |
| Demo | 25% | 3 tabs, demo mode, the "proof by failing test" moment |

## 6. Cut list if behind schedule (cut in this order)
1. PBO (keep DSR)
2. Cost stress
3. Streamlit polish → plain CLI + recorded terminal
4. **Never cut:** sealed holdout, notebook-mode proof tests, demo video
