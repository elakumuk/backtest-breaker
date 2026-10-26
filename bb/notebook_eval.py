"""Evaluate notebook detectors against a planted-truth fixture set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "examples" / "notebooks"
LABELS_PATH = NOTEBOOK_DIR / "labels.json"


def score(
    findings_by_notebook: dict[str, list[dict[str, Any]]],
    labels: dict[str, list[dict[str, Any]]],
) -> tuple[float, float, int]:
    """Return precision, recall, and false-PROVEN count on clean.ipynb.

    Only findings marked PROVEN count as predictions. A match requires the same
    bug class and a cell index within one cell of a ground-truth label.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    clean_false_proven = 0
    for notebook, truth in labels.items():
        predicted = [finding for finding in findings_by_notebook.get(notebook, []) if finding.get("status") == "PROVEN"]
        unmatched = list(truth)
        for finding in predicted:
            index = finding.get("cell_index")
            bug_class = finding.get("bug_class")
            match = next(
                (
                    label for label in unmatched
                    if label.get("bug_class") == bug_class
                    and isinstance(index, int)
                    and abs(index - int(label["cell_index"])) <= 1
                ),
                None,
            )
            if match is None:
                false_positives += 1
                if Path(notebook).name == "clean.ipynb":
                    clean_false_proven += 1
            else:
                true_positives += 1
                unmatched.remove(match)
        false_negatives += len(unmatched)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    return precision, recall, clean_false_proven


def _counts(findings: list[dict[str, Any]], labels: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    """Compute per-notebook TP, FP, FN, and PROVEN counts for display."""
    proven = [finding for finding in findings if finding.get("status") == "PROVEN"]
    unmatched = list(labels)
    tp = fp = 0
    for finding in proven:
        match = next((
            label for label in unmatched
            if finding.get("bug_class") == label.get("bug_class")
            and isinstance(finding.get("cell_index"), int)
            and abs(finding["cell_index"] - int(label["cell_index"])) <= 1
        ), None)
        if match is None:
            fp += 1
        else:
            tp += 1
            unmatched.remove(match)
    return tp, fp, len(unmatched), len(proven)


def main() -> None:
    """Run each detector on the planted notebooks and print evaluation counts."""
    from bb.notebook_mode import LLMDetector, StaticDetector, audit_notebook

    labels: dict[str, list[dict[str, Any]]] = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    notebooks = sorted(labels)
    detectors = [StaticDetector(), LLMDetector()]
    print("detector | notebook | TP | FP | FN | PROVEN")
    for detector in detectors:
        all_findings: dict[str, list[dict[str, Any]]] = {}
        for filename in notebooks:
            report = audit_notebook(NOTEBOOK_DIR / filename, detector=detector, clean_notebook=NOTEBOOK_DIR / "clean.ipynb")
            findings = report.findings
            all_findings[filename] = findings
            tp, fp, fn, proven = _counts(findings, labels[filename])
            print(f"{detector.name} | {filename} | {tp} | {fp} | {fn} | {proven}")
        precision, recall, false_clean = score(all_findings, labels)
        print(f"{detector.name} total | precision={precision:.3f} | recall={recall:.3f} | clean false-PROVEN={false_clean}")


if __name__ == "__main__":
    main()
