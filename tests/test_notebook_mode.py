"""Eval-first tests for notebook detectors and proof-specificity safeguards."""

from pathlib import Path

import pytest

from bb.notebook_mode import StaticDetector, audit_notebook

NOTEBOOKS = Path(__file__).resolve().parents[1] / "examples" / "notebooks"


@pytest.mark.parametrize(
    ("notebook", "bug_class"),
    [
        ("leaky_shift.ipynb", "lookahead_shift"),
        ("leaky_norm.ipynb", "fullsample_leak"),
        ("leaky_universe.ipynb", "universe_selection"),
    ],
)
def test_planted_leaks_are_detected_and_proven(notebook, bug_class):
    """Every planted static finding fails only on its own notebook."""
    report = audit_notebook(
        NOTEBOOKS / notebook,
        detector=StaticDetector(),
        clean_notebook=NOTEBOOKS / "clean.ipynb",
        timeout_seconds=15,
    )
    assert len(report.findings) == 1
    assert report.findings[0]["bug_class"] == bug_class
    assert report.findings[0]["status"] == "PROVEN"
    assert "INVARIANT_VIOLATED" in report.findings[0]["runner_stdout_tail"]["candidate"]
    assert "passed" in report.findings[0]["runner_stdout_tail"]["clean_sanity"]


def test_clean_notebook_has_no_proven_findings():
    """The clean reference produces no detector findings."""
    report = audit_notebook(NOTEBOOKS / "clean.ipynb", detector=StaticDetector())
    assert sum(item["status"] == "PROVEN" for item in report.findings) == 0


def test_finding_that_also_fails_on_clean_is_downgraded():
    """A test is not accepted as proof if the supplied clean control fails too."""
    report = audit_notebook(
        NOTEBOOKS / "leaky_shift.ipynb",
        detector=StaticDetector(),
        clean_notebook=NOTEBOOKS / "leaky_shift.ipynb",
        timeout_seconds=15,
    )
    assert report.findings[0]["status"] == "UNPROVEN"
    assert "not specific" in report.findings[0]["reason"]


def test_runner_timeout_is_reported_as_unproven(tmp_path, monkeypatch):
    """A deliberately sleeping generated test is killed and never called proof."""
    from bb import notebook_mode

    slow_code = "import time\ndef test_slow():\n    time.sleep(60)\n"
    monkeypatch.setattr(notebook_mode, "_template_test_code", lambda bug_class: slow_code)
    report = audit_notebook(
        NOTEBOOKS / "leaky_shift.ipynb",
        detector=StaticDetector(),
        clean_notebook=NOTEBOOKS / "clean.ipynb",
        timeout_seconds=1.5,
    )
    finding = report.findings[0]
    assert finding["status"] == "UNPROVEN"
    assert finding["reason"] == "candidate test timed out"
    assert "RUNNER_TIMEOUT" in finding["runner_stdout_tail"]["candidate"]


def test_wrong_cell_fix_is_unproven():
    class Detector(StaticDetector):
        name = "test"
        def detect(self, notebook_path):
            finding = super().detect(notebook_path)[0]
            finding["fix"] = {"cell_index": 0, "old": "signal(prices)", "new": "signal(prices).shift(1)"}
            return [finding]
    result = audit_notebook(NOTEBOOKS / "leaky_shift.ipynb", detector=Detector(), clean_notebook=NOTEBOOKS / "clean.ipynb")
    assert result.findings[0]["status"] == "UNPROVEN"


def test_fix_old_must_occur_once():
    class Detector(StaticDetector):
        name = "test"
        def detect(self, notebook_path):
            finding = super().detect(notebook_path)[0]
            finding["fix"] = {"cell_index": finding["cell_index"], "old": "missing source", "new": "x"}
            return [finding]
    result = audit_notebook(NOTEBOOKS / "leaky_shift.ipynb", detector=Detector(), clean_notebook=NOTEBOOKS / "clean.ipynb")
    assert result.findings[0]["status"] == "UNPROVEN"


def test_same_fix_is_deduped_by_severity():
    class Detector(StaticDetector):
        name = "test"
        def detect(self, notebook_path):
            finding = super().detect(notebook_path)[0]
            duplicate = dict(finding, severity="low")
            return [finding, duplicate]
    result = audit_notebook(NOTEBOOKS / "leaky_shift.ipynb", detector=Detector(), clean_notebook=NOTEBOOKS / "clean.ipynb")
    assert sorted(item["status"] for item in result.findings) == ["DUPLICATE", "PROVEN"]


def test_false_positive_hypothesis_is_kept_in_dismissed():
    class Detector(StaticDetector):
        name = "test"
        def detect(self, notebook_path):
            return [{"cell_index": 1, "line": 1, "bug_class": "no_costs", "explanation": "Costs are modeled.", "severity": "low", "is_bug": False, "fix": None}]
    result = audit_notebook(NOTEBOOKS / "clean.ipynb", detector=Detector())
    assert result.findings == []
    assert result.dismissed[0]["status"] == "DISMISSED"


def test_schema_failure_is_reported_as_clean_unproven(monkeypatch):
    """An invalid detector response is surfaced as UNPROVEN instead of crashing."""
    from bb import notebook_mode
    from bb.llm import SchemaValidationError

    monkeypatch.setenv("PROVIDER", "baseten")
    monkeypatch.setattr(notebook_mode, "chat", lambda *args, **kwargs: (_ for _ in ()).throw(SchemaValidationError("invalid")))
    report = audit_notebook(NOTEBOOKS / "clean.ipynb", detector=notebook_mode.LLMDetector())
    assert report.findings[0]["status"] == "UNPROVEN"
    assert report.findings[0]["reason"] == "model output did not match schema"
