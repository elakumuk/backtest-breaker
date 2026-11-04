"""Tests for the mock LLM provider."""

import json
from types import SimpleNamespace

import pytest

from bb.llm import SchemaValidationError, _openai_response_format, chat


def test_openai_json_schema_wrapper():
    """OpenAI-compatible providers receive strict named structured output."""
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    assert _openai_response_format(schema) == {
        "type": "json_schema",
        "json_schema": {"name": "output", "schema": schema, "strict": True},
    }


def test_mock_returns_schema_valid_json(monkeypatch):
    """Mock output follows required object properties and simple types."""
    monkeypatch.setenv("PROVIDER", "mock")
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}, "name": {"type": "string"}},
        "required": ["count", "name"],
    }
    response = chat([], json_schema=schema)
    value = json.loads(response.content)
    assert isinstance(value["count"], int)
    assert isinstance(value["name"], str)


@pytest.mark.parametrize("bad_fix", [
    {"from": "x", "to": "y"},
    {"old_code": "x", "new_code": "y"},
    {"original": "x", "replacement": "y"},
    {"minimal": "replace it", "evidence": "x"},
])
def test_structured_output_validation_retries_bad_fix_once(monkeypatch, bad_fix):
    """Provider responses using unsupported nested fix keys are retried once."""
    from openai import OpenAI

    class Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            item = (json.dumps({"findings": [{
                "cell_index": 1, "line": 2, "bug_class": "lookahead_shift",
                "explanation": "leak", "severity": "high", "is_bug": True,
                "fix": bad_fix,
            }]}) if len(self.calls) == 1 else json.dumps({"findings": [{
                "cell_index": 1, "line": 2, "bug_class": "lookahead_shift",
                "explanation": "leak", "severity": "high", "is_bug": True,
                "fix_old": "x", "fix_new": "x.shift(1)",
            }]}))
            message = SimpleNamespace(content=item, tool_calls=[])
            return SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
            )

    completions = Completions()
    monkeypatch.setenv("PROVIDER", "baseten")
    monkeypatch.setenv("BASETEN_API_KEY", "test")
    monkeypatch.setenv("BASETEN_MODEL", "test-model")
    monkeypatch.setenv("BASETEN_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(OpenAI, "__init__", lambda self, **kwargs: setattr(self, "chat", SimpleNamespace(completions=completions)))
    response = chat([], json_schema={
        "type": "object",
        "properties": {"findings": {"type": "array", "items": {
            "type": "object", "properties": {"fix_old": {"type": "string"}, "fix_new": {"type": "string"}},
            "required": ["fix_old", "fix_new"], "additionalProperties": True,
        }}},
        "required": ["findings"], "additionalProperties": False,
    })
    assert len(completions.calls) == 2
    assert "validation" in completions.calls[1]["messages"][-1]["content"].lower()
    assert json.loads(response.content)["findings"][0]["fix_old"] == "x"


def test_schema_validation_error_is_typed(monkeypatch):
    """A second invalid structured response raises the typed caller error."""
    from openai import OpenAI

    class Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            message = SimpleNamespace(content='{"wrong": true}', tool_calls=[])
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=message, finish_reason="stop")])

    completions = Completions()
    monkeypatch.setenv("PROVIDER", "baseten")
    monkeypatch.setenv("BASETEN_API_KEY", "test")
    monkeypatch.setenv("BASETEN_MODEL", "test-model")
    monkeypatch.setenv("BASETEN_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(OpenAI, "__init__", lambda self, **kwargs: setattr(self, "chat", SimpleNamespace(completions=completions)))
    with pytest.raises(SchemaValidationError):
        chat([], json_schema={"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}})
    assert completions.calls == 2


def test_tool_call_turn_skips_schema_validation(monkeypatch):
    """A tool-call turn has empty content and must not be schema-validated (live auditor regression)."""
    from openai import OpenAI

    class Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            call = SimpleNamespace(id="c1", function=SimpleNamespace(name="walk_forward", arguments="{}"))
            message = SimpleNamespace(content=None, tool_calls=[call])
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=message, finish_reason="tool_calls")])

    completions = Completions()
    monkeypatch.setenv("PROVIDER", "baseten")
    monkeypatch.setenv("BASETEN_API_KEY", "test")
    monkeypatch.setenv("BASETEN_MODEL", "test-model")
    monkeypatch.setenv("BASETEN_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(OpenAI, "__init__", lambda self, **kwargs: setattr(self, "chat", SimpleNamespace(completions=completions)))
    response = chat([], tools=[{"type": "function"}], json_schema={"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}})
    assert completions.calls == 1
    assert response.tool_calls[0]["name"] == "walk_forward"
