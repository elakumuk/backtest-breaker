# Backtest Breaker

**An AI that tries to break your backtest before the market does.**

Built from scratch during the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026) · Indie Developer track · Theme: *Build for Your World*.
Runtime model: **Muse Spark** (`meta-models/muse-spark-open-1026`, served through Baseten).

> Backtest Breaker audits trading strategies. It does not recommend trades or give investment advice.

---

## The problem

Most backtests that look great are wrong, and their authors can't see why. Two failure modes cause most of it:

1. **Selection through iteration.** You try 25 variants, keep the best, and report its Sharpe as if you had only tried one.
2. **Look-ahead leakage.** A `bfill`, a centered rolling window, or a scaler fit on the full sample quietly uses future data.

AI research assistants make both faster.

## What it does

Two Muse Spark agents with opposite jobs. Everything numeric is plain, deterministic code.

```
Research agent (Muse Spark)   proposes StrategySpec JSON, iterates on IN-SAMPLE data only
          │
DataGate + deterministic engine   signal(t) uses data ≤ t-1; the holdout is hash-sealed
          │
Audit battery (code)          walk-forward · Deflated Sharpe · PBO (CSCV) · parameter sensitivity · cost stress
          │
Auditor agent (Muse Spark)    tool-calls the attacks, logs a PREDICTION, opens the sealed holdout ONCE
          │
Code rubric                   final verdict: SURVIVES / FRAGILE / BROKEN (the model's opinion is shown, not trusted)
```

**Notebook mode:** Muse Spark reads a `.ipynb`, names the leaky cell and line, and proposes a minimal fix. A finding is **PROVEN** only if a generated invariant test fails on the original code, passes with the fix applied, and passes on a clean control notebook.

## What happened when we ran it live

Every number below is read from `demo/MANIFEST.json`, which `scripts/build_manifest.py` builds from the saved run files.

We gave Muse Spark 25 in-sample trials and told it to improve Sharpe. **It mostly declined to overfit.** It named its specs things like *"NOT Optimized for In-Sample Sharpe"* and, from trial 14 onward, raised its own cost assumption from 20 to 100 bps. The one exception was **trial 13, where it set trading costs to zero**, and that trial became the in-sample "best."

| | |
|---|---|
| Trials (live Muse Spark, 0 failed) | 25 |
| Median trial Sharpe (in-sample) | 0.56 |
| Best in-sample Sharpe (trial 13, `cost_bps = 0`) | 0.68 |
| Same strategy at a realistic 10 bps | 0.65 |
| Auditor's **logged prediction** for the holdout, before it opened | 0.32 |
| **Actual sealed-holdout Sharpe** (2024 →, opened once, hash-verified) | 0.30 |
| Deflated Sharpe ratio | 0.04 |
| Code-rubric verdict | **FRAGILE** (DSR < 0.95; holdout < half of in-sample) |
| Muse Spark's own verdict | SURVIVES. The rubric overrules it; the disagreement is shown. |

Three things we didn't expect: the model refused most of the overfitting we asked for; its holdout prediction (0.32) landed close to the truth (0.30); and it still called the strategy "SURVIVES." That last one is why the verdict belongs to code.

**What the auditor did *not* catch:** it ran the cost-stress test, but it never said explicitly that the winner was selected at zero cost. That's the next rubric rule (see Limitations).

### Held-out notebook eval

5 leaky notebooks (bfill over gaps, centered rolling window, scaler fit on all rows, shuffled split, `merge_asof` forward), 5 clean twins, and 2 decoys. They were written outside this repo, so the build agent could never tune to them.

| Detector | Result |
|---|---|
| Regex baseline (ours) | Early score of P 0.62 / R 1.00 was fake: it matched our own example variable names. With proof-by-fix: **0 / 0**. The held-out set caught our own baseline gaming our own eval. |
| Muse Spark detection | 5/5 real bugs at the right cell in the earlier runs (Oct 26, Oct 30); 4/5 in the latest run. 0 findings on clean twins and decoys in every run (not fooled by a `shift(-1)` used only in a diagnostic). |
| Muse Spark **PROVEN**, latest run (after the schema fix) | **Precision 1.00 / recall 0.60**: 3 of 5 bugs proven by fix + clean control, 0 false PROVEN on 7 clean/decoy notebooks. |

The gap between "found" and "PROVEN" is deliberate. We would rather leave a real bug UNPROVEN than mark a clean notebook as broken.

## Quick start

```bash
git clone https://github.com/elakumuk/backtest-breaker && cd backtest-breaker
pip install -r requirements.txt
streamlit run app/streamlit_app.py     # Demo replay: saved live runs, no network, no API key
pytest
```

Live mode (Baseten key with Muse Spark access):

```bash
cp .env.example .env     # PROVIDER=baseten, BASETEN_API_KEY, BASETEN_BASE_URL, BASETEN_MODEL
python -m bb.agents.research --iters 25                  # research agent, in-sample only
python -m bb.agents.auditor --run-dir runs/<run-id>      # auditor + one-shot sealed holdout
python -m bb.notebook_mode path/to/notebook.ipynb --clean path/to/clean.ipynb
```

Data: Ken French Data Library industry portfolios, downloaded at runtime and never committed. `demo/` holds only our own derived strategy returns and reports.

## Design decisions

| Decision | Why |
|---|---|
| The LLM never computes a metric | In probes Muse Spark rounded or invented numbers (1.9 → 1.92). Every number in its explanation is checked against tool output (`grounded`). |
| The verdict comes from a code rubric | The model can disagree (`model_disagrees`), and here it did. |
| `DataGate` seals the holdout structurally | The research agent can't read holdout rows. The auditor must register a prediction before the vault opens, and it opens once. |
| Proof by fix + clean control | "The model says it's a bug" isn't evidence. A test that fails on the bug, passes on the fix, and passes on clean code is. |
| Client-side schema validation + one retry | The provider did not enforce strict nested schemas. We validate every structured reply ourselves. |

## How Meta tooling is used

- **Muse Spark as research agent:** structured `StrategySpec` proposals from in-sample feedback.
- **Muse Spark as auditor:** tool calling (walk-forward, sensitivity, cost stress, DSR, PBO, holdout), plus a written holdout prediction.
- **Muse Spark in notebook mode:** localizes the leak, writes the fix, and the proof test is generated from it.

## Limitations

- No rule yet flags "selected at zero cost." Cost stress reports it, but the rubric doesn't act on it.
- One asset universe (Ken French industry portfolios) and simple long/short factor-style specs.
- One holdout period (2024 onward). A sealed one-shot holdout is a strong check, not a guarantee.
- Muse Spark is a reasoning model (6–60+ s per call), so the app defaults to replaying saved live runs.

## Built during the event

`BUILD_LOG.md` (dated, per day, which tool did what) and the commit history start on Oct 26. Development used AI coding assistants (Codex, Claude), logged honestly; the product's runtime AI is Meta-only. MIT License.
