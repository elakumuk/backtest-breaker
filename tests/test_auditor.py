"""Offline tests for auditor decision rules and the tool-call mock contract."""
import json

from bb.agents.auditor import _reasons_and_verdict, _ungrounded_numbers
from bb.llm import chat


def test_rubric_broken_holdout_and_cost():
    verdict, reasons = _reasons_and_verdict({"open_holdout":{"sharpe":-0.1}}, 1.0)
    assert verdict == "BROKEN" and any("holdout" in r["rule"] for r in reasons)
    verdict, _ = _reasons_and_verdict({"cost_stress":{"25":-0.1}}, 1.0)
    assert verdict == "BROKEN"


def test_rubric_broken_dsr_and_pbo():
    verdict, _ = _reasons_and_verdict({"deflated_sharpe":{"dsr":0.4},"pbo":{"pbo":0.6}}, 1.0)
    assert verdict == "BROKEN"


def test_rubric_fragile_flags_and_holdout_cap():
    verdict, reasons = _reasons_and_verdict({"walk_forward":{"median_sharpe":0.4},"param_sensitivity":{"spread":0.6},"deflated_sharpe":{"dsr":0.9},"pbo":{"pbo":0.4}}, 1.0)
    assert verdict == "FRAGILE" and len(reasons) >= 4
    assert _reasons_and_verdict({}, 1.0)[0] == "FRAGILE"


def test_rubric_survives():
    verdict, reasons = _reasons_and_verdict({"open_holdout":{"sharpe":1.2},"cost_stress":{"25":0.5},"walk_forward":{"median_sharpe":0.8},"param_sensitivity":{"spread":0.2},"deflated_sharpe":{"dsr":0.99},"pbo":{"pbo":0.1}}, 1.0)
    assert verdict == "SURVIVES" and reasons == []


def test_mock_auditor_returns_scripted_tool_calls(monkeypatch):
    monkeypatch.setenv("PROVIDER", "mock")
    response=chat([], tools=[{"type":"function"}], mock_policy={"kind":"auditor","turn":0,"script":[{"tool_calls":[{"id":"x","name":"cost_stress","arguments":{}}]}]})
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == [{"id":"x","name":"cost_stress","arguments":{}}]


def test_number_grounding_allowed_precision_and_percentages():
    evidence={"values":[1.9012,-0.2034,0.25]}
    assert _ungrounded_numbers(["Sharpe fell from 1.92 to -0.21"],evidence,set()) == ["1.92","-0.21"]
    assert _ungrounded_numbers(["Sharpe was 1.90 and return was 25%"],evidence,set()) == []
    assert _ungrounded_numbers(["Reported value 0.684"],{"x":0.6838},set()) == ["0.684"]
