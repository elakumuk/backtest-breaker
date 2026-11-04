# Held-out Notebook Mode eval — 2026-10-26 (pre-D6 code, commit 26601cf)

Set: 5 leaky notebooks (bfill over gaps, centered rolling window, StandardScaler fit on all
rows, shuffled train_test_split, merge_asof direction="forward") + 5 clean twins + 2 decoys
(shift(-1) used only in an IC diagnostic; fixed pre-set vol target). Bug is always in cell 2.
Ground truth verified by `run_heldout.py verify` (perturb rows >= c, signal rows <= c must not move).

| Detector | Scored as-is (+/-1 cell, any PROVEN) | What really happened |
|---|---|---|
| StaticDetector | P 0.62 / R 1.00 | Flags cell 3 (backtest) on EVERY notebook incl. clean. Exact-cell recall = **0/5**. Its "TPs" are the non-specific proof + the +/-1 tolerance. |
| LLMDetector (Muse Spark, Baseten) | P 0.29 / R 1.00 | Found all **5/5** real bugs at the right cell + line with correct explanations; **0** real claims on clean twins/decoys (wasn't fooled by the shift(-1) decoy). Low precision = schema forced "No bug" findings for every class, and the non-specific proof marked them PROVEN. Excluding "No bug" findings: ~5/6 (one duplicate). |

Latency: 31-252 s per notebook (reasoning model) -> demo must replay saved runs.

Fixes handed to Codex in CODEX_D6 Part A: `is_bug` field, proof-by-fix, dedupe, exact-cell scoring.
Re-run `python run_heldout.py llm` after D6 and compare. Commit this folder to the repo only at D8.

Demo line: "Our own baseline gamed our own eval. A held-out set caught it, and the fix made
every PROVEN finding prove both the bug and its one-line fix."

---

# Re-run after D6 — 2026-10-30 (commit b3ef860)

| Detector | P / R (PROVEN only) | What really happened |
|---|---|---|
| StaticDetector | 0.00 / 0.00 | Proof-by-fix killed the fake 0.62/1.00: its fixes are hardcoded to Codex's own example variable names, so none apply. Honest baseline. |
| LLMDetector (Muse Spark) | 0.00 / 0.00 | **Detection is now ~perfect**: exactly 5 raw findings, 5/5 at the right cell, 0 on clean twins, 0 on both decoys (was 44 raw findings). All 5 UNPROVEN only because the model ignored the nested `fix` keys (from/to, old_code/new_code, original/replacement, prose). Provider-side strict schema not enforced. |

Latency dropped 31-252 s -> 6-30 s per notebook (no more one-finding-per-class padding).

Fix handed to Codex in CODEX_D7 Part 0: flatten to fix_old/fix_new + client-side jsonschema validation + 1 retry.
Expected after fix: P/R ~1.00 if the model's fix strings apply verbatim. Re-run `run_heldout.py llm` after D7.

---

# Re-run after D7 Part 0 schema fix — 2026-11-03 (fix_old/fix_new + jsonschema + 1 retry)

LLMDetector (Muse Spark), PROVEN only: **P 1.00 / R 0.60** (TP 3, FP 0, FN 2; raw findings 4).
- PROVEN: centered_rolling, shuffled_split, asof_forward.
- scaler_fit_all: found at the right cell but UNPROVEN (proof did not pass all three checks).
- bfill_gaps: not detected this run (it was detected on 10-26 and 10-30). Detection varies run to run.
- 0 findings on all 5 clean twins and both decoys.
Latency 5-63 s per notebook.
