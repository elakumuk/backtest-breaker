"""Provider-agnostic chat entry point; runtime providers are Meta models only."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from jsonschema import ValidationError, validate

logger = logging.getLogger(__name__)
load_dotenv()


class SchemaValidationError(ValueError):
    """A structured model response remained invalid after one correction retry."""


def _validate_content(content: str, schema: dict[str, Any]) -> None:
    """Parse and validate a JSON response against its requested schema."""
    try:
        payload = json.loads(content)
        validate(instance=payload, schema=schema)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise SchemaValidationError(str(exc)) from exc


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response returned by :func:`chat`."""

    content: str
    model: str
    provider: str
    usage: dict[str, int]
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    arguments_error: str | None = None


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
        return [_schema_example(schema.get("items", {})) for _ in range(int(schema.get("minItems", 1)))]
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


def _mock_policy_response(mock_policy: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic structured output for offline research-agent runs."""
    policy = mock_policy.get("kind")
    if policy == "prediction":
        return {
            "predicted_holdout_sharpe": float(mock_policy.get("predicted_sharpe", 0.0)),
            "rationale": "Mock prediction based on the in-sample trial history.",
        }
    if policy == "auditor":
        script = mock_policy.get("script", [])
        turn = int(mock_policy.get("turn", 0))
        if turn < len(script):
            item = script[turn]
            if "tool_calls" in item:
                return {"_tool_calls": item["tool_calls"], "_finish_reason": "tool_calls"}
            return item
        return {"model_verdict": "FRAGILE", "bullets": ["The audit completed using computed evidence."], "predicted_holdout_sharpe": 0.0}
    if policy != "researcher":
        raise ValueError(f"Unknown mock policy: {policy}")
    rng = random.Random(int(mock_policy.get("seed", 0)) + 1_000_003 * int(mock_policy.get("iteration", 0)))
    best = mock_policy.get("best_spec")
    if not best:
        candidate = {
            "name": "mock momentum candidate", "signal_type": "momentum",
            "lookback_days": 126, "skip_days": 21, "top_n": 5,
            "rebalance": "M", "long_short": False, "cost_bps": 10.0,
            "start": mock_policy["start"], "end": mock_policy["end"],
        }
    else:
        candidate = dict(best)
        candidate["lookback_days"] = max(5, min(504, candidate["lookback_days"] + rng.choice([-42, -21, 21, 42])))
        candidate["top_n"] = max(1, min(24, candidate["top_n"] + rng.choice([-2, -1, 1, 2])))
        if rng.random() < 0.25:
            choices = [item for item in ("momentum", "reversal", "low_vol", "trend") if item != candidate["signal_type"]]
            candidate["signal_type"] = rng.choice(choices)
        candidate["name"] = "mock hill-climb candidate"
        candidate["start"] = mock_policy["start"]
        candidate["end"] = mock_policy["end"]
    return candidate


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    json_schema: dict[str, Any] | None = None,
    mock_policy: dict[str, Any] | None = None,
) -> LLMResponse:
    """Send a chat request through the configured provider, retrying transient errors."""
    provider = os.getenv("PROVIDER", "mock").strip().lower()
    if os.getenv("BB_DEMO_MODE") == "1" and provider != "mock":
        raise RuntimeError("Demo mode is offline: real LLM providers are disabled")
    if provider not in {"mock", "meta", "baseten", "fireworks"}:
        raise ValueError(f"Unsupported PROVIDER: {provider}")
    model = os.getenv(f"{provider.upper()}_MODEL", "mock-v1") if provider != "mock" else "mock-v1"
    if provider == "mock":
        if mock_policy is not None:
            payload = _mock_policy_response(mock_policy)
            calls = payload.pop("_tool_calls", [])
            finish_reason = payload.pop("_finish_reason", "stop")
            content = json.dumps(payload, separators=(",", ":")) if payload else ""
        elif json_schema is not None:
            payload = _schema_example(json_schema)
            content = json.dumps(payload, separators=(",", ":"))
        else:
            prompt = next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "user"), "")
            content = "Mock response" if not prompt else f"Mock response: {prompt[:120]}"
        response = LLMResponse(content, model, provider, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, calls if mock_policy and mock_policy.get("kind") == "auditor" else [], finish_reason if mock_policy and mock_policy.get("kind") == "auditor" else "stop")
        if json_schema is not None and not response.tool_calls:
            _validate_content(response.content, json_schema)
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
    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=180)
    request: dict[str, Any] = {"model": model, "messages": messages}
    if json_schema is not None:
        request["response_format"] = _openai_response_format(json_schema)
    if tools is not None:
        request["tools"] = tools
    last_error: Exception | None = None
    validation_retried = False
    for attempt in range(3):
        try:
            result = client.chat.completions.create(**request)
            usage = result.usage.model_dump() if result.usage else {}
            message = result.choices[0].message
            content = message.content or ""
            # A tool-call turn carries no final JSON yet; validate only final answers.
            if json_schema is not None and not message.tool_calls:
                try:
                    _validate_content(content, json_schema)
                except SchemaValidationError as exc:
                    if validation_retried:
                        raise
                    validation_retried = True
                    request["messages"] = [
                        *messages,
                        {"role": "user", "content": f"Your previous JSON failed client-side schema validation: {exc}. Return corrected JSON matching the schema exactly."},
                    ]
                    continue
            calls = []
            arguments_error = None
            for call in (message.tool_calls or []):
                raw = call.function.arguments
                try:
                    arguments = json.loads(raw)
                except (TypeError, json.JSONDecodeError) as exc:
                    arguments = raw
                    arguments_error = str(exc)
                calls.append({"id": call.id, "name": call.function.name, "arguments": arguments})
            logger.info("llm provider=%s model=%s usage=%s", provider, model, usage)
            return LLMResponse(content, model, provider, usage, calls, result.choices[0].finish_reason, arguments_error)
        except SchemaValidationError:
            raise
        except Exception as exc:  # SDK errors may include transient network/server faults.
            last_error = exc
            if attempt == 2:
                break
            time.sleep(0.5 * (2**attempt))
    assert last_error is not None
    raise last_error
