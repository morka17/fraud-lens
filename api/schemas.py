"""Pydantic schemas for the FraudLens scoring API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ScoreRequest(BaseModel):
    """Incoming transaction to be scored in real time."""

    card_id: str = Field(..., min_length=1, examples=["card_0042"])
    amount: float = Field(..., gt=0, examples=[1250.00])
    merchant: str = Field(..., min_length=1, examples=["Best Buy #4021"])
    merchant_category: str = Field(default="unknown", examples=["electronics_store"])

    @field_validator("amount")
    @classmethod
    def amount_must_be_reasonable(cls, v: float) -> float:
        if v > 1_000_000:
            raise ValueError("amount exceeds sane transaction ceiling")
        return v


class ScoreResponse(BaseModel):
    """Result of scoring a transaction against the current feature vector
    and the deployed XGBoost model."""

    card_id: str
    fraud_score: float = Field(..., ge=0, le=1, description="Model probability of fraud, 0-1.")
    decision: Literal["ALLOW", "FLAG", "BLOCK"]
    latency_ms: float
    feature_source: Literal["online_store", "cold_start_default"] = Field(
        ...,
        description=(
            "Whether features came from Redis (normal path) or a neutral "
            "default vector because this card had no prior activity (cold start)."
        ),
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    redis_connected: bool
    model_loaded: bool