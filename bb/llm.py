"""Provider-agnostic chat entry point; runtime providers are Meta models only."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response returned by :func:`chat`."""

    content: str
    model: str
    provider: str
    usage: dict[str, int]


def _schema_example(schema: dict[str, Any]) -> Any:
    """Build a deterministic, structurally valid example from a JSON schema."""
    if "default" in schema:
        return schema["default"]
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    typ = schema.get("type")
    if typ == "object":
        props = schema.get("properties", {})
        required = schema.get("required", list(props))
        return {key: _schema_example(props[key]) for key in required if key in props}
    if typ == "array":
        return [_schema_example(schema.get("items", {}))]
    if typ == "string":
        return "mock"
    if typ == "integer":
        return schema.get("minimum", 0)
    if typ == "number":
        return schema.get("minimum", 0)
    if typ == "boolean":
        return False
    if typ == "null":
        return None
    for option in schema.get("anyOf", schema.get("oneOf", [])):
        return _schema_example(option)
    return None


def _openai_response_format(schema: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON schema in the strict OpenAI-compatible response format."""
    return {"type": "json_schema", "json_schema": {"name": "output", "schema": schema, "strict": True}}


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    json_schema: dict[str, Any] | None = None,
) -> LLMResponse:
    """Send a chat request through the configured provider, retrying transient errors."""
    del tools  # Tool dispatch will be added with the agent work.
    provider = os.getenv("PROVIDER", "mock").strip().lower()
    if provider not in {"mock", "meta", "baseten", "fireworks"}:
        raise ValueError(f"Unsupported PROVIDER: {provider}")
    model = os.getenv(f"{provider.upper()}_MODEL", "mock-v1") if provider != "mock" else "mock-v1"
    if provider == "mock":
        if json_schema is not None:
            payload = _schema_example(json_schema)
            content = json.dumps(payload, separators=(",", ":"))
        else:
            prompt = next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), "")
            content = "Mock response" if not prompt else f"Mock response: {prompt[:120]}"
        response = LLMResponse(content, model, provider, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        logger.info("llm provider=%s model=%s usage=%s", provider, model, response.usage)
        return response
    if provider == "meta":
        raise NotImplementedError("fill in once Meta Model API docs/credits are available")

    from openai import OpenAI

    prefix = provider.upper()
    api_key = os.getenv(f"{prefix}_API_KEY")
    base_url = os.getenv(f"{prefix}_BASE_URL") if provider == "baseten" else "https://api.fireworks.ai/inference/v1"
    if not api_key or not model or not base_url:
        raise ValueError(f"{prefix}_API_KEY, {prefix}_MODEL, and endpoint configuration are required")
    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    request: dict[str, Any] = {"model": model, "messages": messages}
    if json_schema is not None:
        request["response_format"] = _openai_response_format(json_schema)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = client.chat.completions.create(**request)
            usage = result.usage.model_dump() if result.usage else {}
            content = result.choices[0].message.content or ""
            logger.info("llm provider=%s model=%s usage=%s", provider, model, usage)
            return LLMResponse(content, model, provider, usage)
        except Exception as exc:  # SDK errors may include transient network/server faults.
            last_error = exc
            if attempt == 2:
                break
            time.sleep(0.5 * (2**attempt))
    assert last_error is not None
    raise last_error
