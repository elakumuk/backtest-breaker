"""Score Notebook Mode on the held-out set (outside the repo, so nothing gets tuned to it).

    python run_heldout.py verify            # ground truth: leaky really leaks, clean really doesn't
    python run_heldout.py static            # StaticDetector baseline
    python run_heldout.py llm               # LLMDetector on the provider in the repo .env (Muse Spark)

Scoring:
  * TP  = PROVEN finding on the labeled cell (+/-1) with an accepted bug class
  * FP  = any other PROVEN finding (on a clean twin or decoy this is a false alarm)
  * raw = all findings before proof, so we can see what the proof step filtered out
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from bb.notebook_mode import LLMDetector, StaticDetector, _convert_notebook, audit_notebook  # noqa: E402

HERE = Path(__file__).resolve().parent
NB = HERE / "notebooks"
MANIFEST = json.loads((HERE / "labels.json").read_text())


def _load(nb_path: Path):
    tmp = Path(tempfile.mkdtemp(prefix="heldout-"))
    module_path = _convert_notebook(nb_path, tmp, "m_" + nb_path.stem.replace("-", "_"))
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def leaks(nb_path: Path) -> list[int]:
    """Cutoffs c where perturbing data at rows >= c changes signal rows <= c (should be empty when clean)."""
    m = _load(nb_path)
    prices = m.synthetic_prices()
    base = np.asarray(m.signal(prices), dtype=float)
    bad = []
    for c in range(25, len(prices) - 5, 5):
        for factor in (1.5, 0.6):
            changed = prices.copy()
            changed.iloc[c:] *= factor
            after = np.asarray(m.signal(changed), dtype=float)
            if not np.allclose(base[: c + 1], after[: c + 1], equal_nan=True, atol=1e-12):
                bad.append(c)
                break
    return bad


def verify() -> None:
    ok = True
    for row in MANIFEST:
        bad = leaks(NB / row["notebook"])
        expected_leak = bool(row["labels"])
        good = bool(bad) == expected_leak
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {row['notebook']:<34} leaky={expected_leak!s:<5} violating cutoffs={bad[:6]}")
    print("ground truth holds" if ok else "GROUND TRUTH BROKEN - fix the notebooks before scoring")


def score(detector_name: str) -> None:
    detector = StaticDetector() if detector_name == "static" else LLMDetector()
    tp = fp = fn = raw_total = 0
    rows = []
    for row in MANIFEST:
        t0 = time.time()
        report = audit_notebook(NB / row["notebook"], detector=detector, clean_notebook=NB / row["control"])
        proven = [f for f in report.findings if f.get("status") == "PROVEN"]
        unmatched = list(row["labels"])
        n_tp = n_fp = 0
        for f in proven:
            hit = next((lab for lab in unmatched
                        if f.get("bug_class") in lab["accept"]
                        and isinstance(f.get("cell_index"), int) and abs(f["cell_index"] - lab["cell_index"]) <= 1), None)
            if hit:
                n_tp += 1
                unmatched.remove(hit)
            else:
                n_fp += 1
        tp, fp, fn, raw_total = tp + n_tp, fp + n_fp, fn + len(unmatched), raw_total + len(report.findings)
        reasons = "; ".join(sorted({f"{f.get('bug_class')}@{f.get('cell_index')}:{f['status']}" for f in report.findings}))
        rows.append({"notebook": row["notebook"], "tp": n_tp, "fp": n_fp, "fn": len(unmatched),
                     "raw": len(report.findings), "seconds": round(time.time() - t0, 1), "findings": report.findings})
        print(f"{row['notebook']:<34} TP={n_tp} FP={n_fp} FN={len(unmatched)} raw={len(report.findings)} "
              f"({time.time() - t0:.0f}s) {reasons}")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"\n{report.detector}: precision={precision:.2f} recall={recall:.2f} "
          f"(TP={tp} FP={fp} FN={fn}, raw findings={raw_total})")
    out = HERE / f"results_{detector_name}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps({"detector": report.detector, "precision": precision, "recall": recall,
                               "tp": tp, "fp": fp, "fn": fn, "rows": rows}, indent=2, default=str))
    print(f"saved {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    verify() if cmd == "verify" else score(cmd)
