"""Detect notebook backtest leaks and prove findings with isolated failing tests."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import nbformat

from bb.llm import chat

logger = logging.getLogger(__name__)
BUG_CLASSES = ("lookahead_shift", "fullsample_leak", "universe_selection", "no_costs", "multiple_testing")
FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cell_index": {"type": "integer"},
                    "line": {"type": "integer"},
                    "bug_class": {"type": "string", "enum": list(BUG_CLASSES)},
                    "explanation": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                },
                "required": ["cell_index", "line", "bug_class", "explanation", "severity"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}
TEST_SCHEMA = {
    "type": "object",
    "properties": {"test_code": {"type": "string"}},
    "required": ["test_code"],
    "additionalProperties": False,
}


@dataclass
class RunResult:
    """Captured result of an isolated generated-test execution."""

    returncode: int | None
    output: str
    timed_out: bool = False


class Runner(Protocol):
    """Interface for running generated tests away from the main process."""

    def run(self, test_path: Path, workdir: Path, timeout_seconds: float = 30) -> RunResult:
        """Execute one generated test within its temporary working directory."""


class LocalSubprocessRunner:
    """Run one test via pytest with a timeout and reduced network environment."""

    def run(self, test_path: Path, workdir: Path, timeout_seconds: float = 30) -> RunResult:
        env = dict(os.environ)
        for key in list(env):
            if "proxy" in key.lower():
                env.pop(key, None)
        env["NO_NETWORK"] = "1"
        env["PYTHONPATH"] = str(workdir)
        relative_test = test_path.relative_to(workdir)
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", str(relative_test)],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            return RunResult(completed.returncode, output, False)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return RunResult(None, stdout + stderr + "\nRUNNER_TIMEOUT", True)


class ModalRunner:
    """Future remote runner placeholder."""

    def run(self, test_path: Path, workdir: Path, timeout_seconds: float = 30) -> RunResult:
        """Raise until Modal credits and access are approved."""
        del test_path, workdir, timeout_seconds
        raise NotImplementedError("enable once Modal credits are approved")


@dataclass
class NotebookReport:
    """JSON-serializable report for one notebook audit."""

    notebook: str
    detector: str
    findings: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe object."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize the report as readable JSON."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class StaticDetector:
    """Small AST-assisted baseline detector for common backtest leakage patterns."""

    name = "StaticDetector"

    def detect(self, notebook_path: str | Path) -> list[dict[str, Any]]:
        """Return rule-based findings with notebook cell indices and source lines."""
        notebook = nbformat.read(notebook_path, as_version=4)
        findings: list[dict[str, Any]] = []
        for cell_index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            source = cell.source
            lines = source.splitlines()
            normalized = "\n".join(lines)
            try:
                ast.parse(source)
            except SyntaxError:
                pass
            if (
                re.search(r"pct_change\s*\(|\breturns\b", normalized)
                and re.search(r"(positions|signal)\s*\*\s*(asset_returns|returns)", normalized)
                and not re.search(r"\.shift\s*\(\s*1\s*\)", normalized)
            ):
                line = _line_number(lines, r"positions\s*=\s*signal\(")
                findings.append(_finding(cell_index, line, "lookahead_shift", "Signal-derived positions use same-period returns without a one-bar shift.", "high"))
            if re.search(r"\.mean\s*\(|\.std\s*\(", normalized):
                for line_idx, line_text in enumerate(lines, start=1):
                    if (
                        re.search(r"\b\w*(?:mean|std)\s*=.*\.(mean|std)\s*\(", line_text)
                        and not re.search(r"rolling|expanding|shift\s*\(", line_text)
                    ):
                        findings.append(_finding(cell_index, line_idx, "fullsample_leak", "Signal normalization uses a full-series statistic that can include future observations.", "high"))
                        break
            if re.search(r"\.prod\s*\(\s*axis\s*=\s*0\s*\)", normalized) and re.search(r">\s*0", normalized):
                line = _line_number(lines, r"prod\s*\(\s*axis\s*=\s*0\s*\)")
                findings.append(_finding(cell_index, line, "universe_selection", "The asset universe is selected using full-period cumulative returns.", "critical"))
        return findings


class LLMDetector:
    """Meta-model notebook detector; mock mode uses the deterministic baseline."""

    name = "LLMDetector"

    def __init__(self) -> None:
        self.mock_used = False
        self._static = StaticDetector()

    def detect(self, notebook_path: str | Path) -> list[dict[str, Any]]:
        """Ask the configured Meta provider to inspect ordered code cells."""
        provider = os.getenv("PROVIDER", "mock").strip().lower()
        if provider == "mock":
            self.mock_used = True
            logger.info("LLMDetector mock mode delegated to StaticDetector")
            return self._static.detect(notebook_path)
        notebook = nbformat.read(notebook_path, as_version=4)
        cells = [
            {"cell_index": idx, "source": cell.source}
            for idx, cell in enumerate(notebook.cells)
            if cell.cell_type == "code"
        ]
        response = chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Audit notebook backtest code for bug classes: " + ", ".join(BUG_CLASSES)
                        + ". Return only JSON matching the supplied findings schema. Cite exact cell and line."
                    ),
                },
                {"role": "user", "content": json.dumps(cells)},
            ],
            json_schema=FINDINGS_SCHEMA,
        )
        parsed = json.loads(response.content)
        findings = parsed.get("findings", [])
        for finding in findings:
            finding["detector"] = self.name
        return findings


def _finding(cell_index: int, line: int, bug_class: str, explanation: str, severity: str) -> dict[str, Any]:
    """Create a normalized detector finding."""
    return {
        "cell_index": cell_index,
        "line": line,
        "bug_class": bug_class,
        "explanation": explanation,
        "severity": severity,
        "detector": "StaticDetector",
    }


def _line_number(lines: list[str], pattern: str) -> int:
    """Find the first matching one-based source line, defaulting to line one."""
    return next((idx for idx, line in enumerate(lines, start=1) if re.search(pattern, line)), 1)


def _convert_notebook(notebook_path: Path, output_dir: Path, module_name: str) -> Path:
    """Convert notebook source to a Python module without executing notebook cells."""
    from nbconvert import PythonExporter

    body, _ = PythonExporter().from_filename(str(notebook_path.resolve()))
    module_path = output_dir / f"{module_name}.py"
    source = body
    kept_lines = [
        line for line in source.splitlines()
        if not any(token in line for token in ("get_ipython().", "display(", "plt.show(", ".plot("))
    ]
    module_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    return module_path


def _template_test_code(bug_class: str) -> str:
    """Create a deterministic future-perturbation invariant test for a bug class."""
    if bug_class not in BUG_CLASSES:
        raise ValueError(f"Unsupported bug class: {bug_class}")
    multiplier = "1.5" if bug_class == "lookahead_shift" else ("1.8" if bug_class == "fullsample_leak" else "0.001")
    change_start = "cutoff" if bug_class == "lookahead_shift" else "cutoff + 1"
    return f'''import numpy as np
import module_under_test as target


def test_future_perturbation_does_not_change_past_signal():
    prices = target.synthetic_prices()
    cutoff = len(prices) // 2
    changed = prices.copy()
    changed.iloc[{change_start}:] *= {multiplier}
    before = target.signal(prices)
    after = target.signal(changed)
    if hasattr(before, "reindex"):
        after = after.reindex(index=before.index, columns=before.columns)
    np.testing.assert_allclose(
        np.asarray(before.iloc[: cutoff + 1]),
        np.asarray(after.iloc[: cutoff + 1]),
        rtol=1e-10,
        atol=1e-12,
        err_msg="INVARIANT_VIOLATED: future data changed past signals",
    )
'''


def _llm_test_code(finding: dict[str, Any], module_source: str) -> str:
    """Ask the configured Meta provider for a test using the shared invariant."""
    response = chat(
        [
            {
                "role": "system",
                "content": (
                    "Write a pytest test as Python source only in JSON field test_code. "
                    "The test must perturb synthetic prices after a cutoff and assert that "
                    "signals/positions at or before the cutoff do not change. The assertion "
                    "message must contain INVARIANT_VIOLATED. Import the audited module as "
                    "module_under_test. Do not use network or filesystem access."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"finding": finding, "module_source": module_source}),
            },
        ],
        json_schema=TEST_SCHEMA,
    )
    return str(json.loads(response.content)["test_code"])


def _tail(output: str, lines: int = 30) -> str:
    """Keep only the last diagnostic lines for the report."""
    return "\n".join(output.splitlines()[-lines:])


def audit_notebook(
    notebook_path: str | Path,
    detector: StaticDetector | LLMDetector | None = None,
    clean_notebook: str | Path | None = None,
    runner: Runner | None = None,
    timeout_seconds: float = 30,
) -> NotebookReport:
    """Detect and prove notebook findings; each proof is also checked on clean code."""
    path = Path(notebook_path).resolve()
    detector = detector or LLMDetector()
    runner = runner or LocalSubprocessRunner()
    clean_path = Path(clean_notebook).resolve() if clean_notebook else path.parent / "clean.ipynb"
    findings = detector.detect(path)
    reported: list[dict[str, Any]] = []
    for finding_number, raw_finding in enumerate(findings, start=1):
        finding = dict(raw_finding)
        finding.setdefault("detector", detector.name)
        workdir = Path(tempfile.mkdtemp(prefix="backtest-breaker-notebook-"))
        tests_dir = workdir / "tests_generated"
        tests_dir.mkdir()
        test_path = tests_dir / f"test_finding_{finding_number}.py"
        status = "UNPROVEN"
        reason = ""
        candidate_tail = ""
        clean_tail = ""
        try:
            candidate_module = _convert_notebook(path, workdir, "module_under_test")
            clean_module = _convert_notebook(clean_path, workdir, "clean_module")
            if isinstance(detector, LLMDetector) and not detector.mock_used:
                code = _llm_test_code(finding, candidate_module.read_text(encoding="utf-8"))
            else:
                code = _template_test_code(finding["bug_class"])
            test_path.write_text(code, encoding="utf-8")
            candidate_result = runner.run(test_path, workdir, timeout_seconds)
            candidate_tail = _tail(candidate_result.output)
            if candidate_result.timed_out:
                reason = "candidate test timed out"
            elif candidate_result.returncode == 0:
                reason = "generated invariant test passed on the candidate"
            elif "INVARIANT_VIOLATED" not in candidate_result.output:
                reason = "candidate test failed for a reason other than the invariant"
            else:
                shutil.copyfile(clean_module, candidate_module)
                clean_result = runner.run(test_path, workdir, timeout_seconds)
                clean_tail = _tail(clean_result.output)
                if clean_result.returncode == 0 and not clean_result.timed_out:
                    status = "PROVEN"
                    reason = "invariant failed on candidate and passed on clean control"
                else:
                    reason = "test not specific: it failed or timed out on clean control"
        except Exception as exc:
            reason = f"proof setup failed: {type(exc).__name__}: {exc}"
        finding.update(
            status=status,
            reason=reason,
            test_file=str(test_path),
            runner_stdout_tail={"candidate": candidate_tail, "clean_sanity": clean_tail},
        )
        reported.append(finding)
    detector_name = detector.name
    if isinstance(detector, LLMDetector) and detector.mock_used:
        detector_name += "(mock->static)"
    return NotebookReport(str(path), detector_name, reported)


def main() -> None:
    """Run one notebook audit from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook")
    parser.add_argument("--clean", default=None)
    args = parser.parse_args()
    report = audit_notebook(args.notebook, clean_notebook=args.clean)
    print(report.to_json())


if __name__ == "__main__":
    main()
