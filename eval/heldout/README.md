# Held-out notebook eval

Written by hand **outside this repository** (Oct 26) so the build agent could never see or tune to it, then committed at the end of the event as evidence.
5 leaky notebooks, 5 clean twins, 2 decoys; `labels.json` is ground truth (verified by `run_heldout.py verify`: perturbing rows ≥ c must not move signal rows ≤ c).
Run: `python eval/heldout/run_heldout.py llm` (needs a live Muse Spark key) or `static`. History of every run is in `RESULTS.md`.
