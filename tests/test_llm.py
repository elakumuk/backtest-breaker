"""Tests for the mock LLM provider."""

import json

from bb.llm import chat


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
