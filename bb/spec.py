"""Validated and serializable trading strategy configuration."""

from __future__ import annotations

import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrategySpec(BaseModel):
    """Parameters for a deterministic cross-sectional strategy audit."""

    model_config = ConfigDict(extra="forbid")

    name: str
    signal_type: Literal["momentum", "reversal", "low_vol", "trend"]
    lookback_days: int = Field(ge=5, le=504)
    skip_days: int = Field(ge=0, le=42)
    top_n: int = Field(ge=1, le=24)
    rebalance: Literal["D", "W", "M"]
    long_short: bool
    cost_bps: float = Field(default=10, ge=0, le=200)
    start: date
    end: date

    @model_validator(mode="after")
    def validate_date_order(self) -> "StrategySpec":
        """Require a non-empty chronological evaluation period."""
        if self.start >= self.end:
            raise ValueError("start must be before end")
        return self

    def to_json(self) -> str:
        """Serialize the spec as JSON using ISO date strings."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, value: str) -> "StrategySpec":
        """Validate and parse a JSON encoded spec."""
        return cls.model_validate_json(value)

    @classmethod
    def json_schema(cls) -> dict:
        """Return the JSON schema used for structured LLM output."""
        return cls.model_json_schema()
