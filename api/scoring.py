"""
The real-time scoring path: Redis lookup → feature vector → XGBoost → score.

This module deliberately contains no business logic about WHAT a feature
means — it only reads the feature vector Redis already has (written by
streaming/sinks.py) using the exact same key format and column list defined
in streaming/features.py. If a feature is renamed there, it is
correct-by-construction here too.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import redis
import xgboost as xgb

from streaming.features import FEATURE_COLUMNS, redis_feature_key

logger = logging.getLogger("fraudlens.scoring")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
MODEL_PATH = os.getenv("MODEL_PATH", "./training/model.pkl")
FRAUD_SCORE_THRESHOLD = float(os.getenv("FRAUD_SCORE_THRESHOLD", "0.75"))
FRAUD_SCORE_BLOCK_THRESHOLD = float(os.getenv("FRAUD_SCORE_BLOCK_THRESHOLD", "0.92"))

# Neutral feature vector used when a card has no prior history in Redis
# (brand-new card, or its feature key TTL'd out). Deliberately mid-range
# rather than all-zeros so a genuinely new card isn't auto-flagged just for
# being new.
COLD_START_DEFAULTS: dict[str, float] = {
    "txn_count_1h": 1.0,
    "txn_amount_sum_1h": 50.0,
    "txn_amount_avg_1h": 50.0,
    "txn_amount_max_1h": 50.0,
    "txn_count_24h": 3.0,
    "txn_amount_avg_24h": 50.0,
    "distinct_merchant_categories_1h": 1.0,
    "amount_zscore_vs_24h": 0.0,
}


@dataclass
class ScoringResult:
    fraud_score: float
    decision: str
    feature_source: str
    latency_ms: float


class FeatureClient:
    """Thin wrapper around Redis reads. Isolated behind an interface so the
    Redis client can be swapped or mocked without touching scoring logic."""

    def __init__(self, client: redis.Redis | None = None) -> None:
        self._client = client or redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
        )

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False

    def get_features(self, card_id: str) -> dict | None:
        """Returns the raw feature dict for a card, or None if not present
        (cold start / expired TTL)."""
        raw = self._client.get(redis_feature_key(card_id))
        if raw is None:
            return None
        return json.loads(raw)


class ModelWrapper:
    """Loads the XGBoost model artifact produced by training/train.py and
    exposes a single predict_proba call over the canonical feature order."""

    def __init__(self, path: str = MODEL_PATH) -> None:
        self.path = path
        self.model: xgb.XGBClassifier | None = None
        self.feature_columns: list[str] = FEATURE_COLUMNS

    def load(self) -> None:
        if not Path(self.path).exists():
            raise FileNotFoundError(
                f"No model found at {self.path}. Run `make train` (python -m training.train) first."
            )
        with open(self.path, "rb") as f:
            artifact = pickle.load(f)
        self.model = artifact["model"]
        self.feature_columns = artifact.get("feature_columns", FEATURE_COLUMNS)
        logger.info("Model loaded from %s (%d features)", self.path, len(self.feature_columns))

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_proba(self, feature_vector: list[float]) -> float:
        if self.model is None:
            raise RuntimeError("Model has not been loaded — call .load() first.")
        proba = self.model.predict_proba([feature_vector])[0][1]
        return float(proba)


def build_feature_vector(features: dict, columns: list[str]) -> list[float]:
    """Orders a raw feature dict (from Redis JSON or cold-start defaults)
    into the exact column order the model was trained on. Missing individual
    fields fall back to the cold-start default for that field so a partially
    populated Redis record never crashes inference.
    """
    return [float(features.get(col, COLD_START_DEFAULTS.get(col, 0.0))) for col in columns]


def decide(fraud_score: float) -> str:
    if fraud_score >= FRAUD_SCORE_BLOCK_THRESHOLD:
        return "BLOCK"
    if fraud_score >= FRAUD_SCORE_THRESHOLD:
        return "FLAG"
    return "ALLOW"


def score_transaction(
    card_id: str,
    feature_client: FeatureClient,
    model: ModelWrapper,
) -> ScoringResult:
    """The end-to-end scoring path called by the /score endpoint. Measures
    its own latency so the API can report it back to the caller, which is
    how the p99 latency numbers in the README are actually verified rather
    than just claimed.
    """
    start = time.perf_counter()

    features = feature_client.get_features(card_id)
    feature_source = "online_store"
    if features is None:
        logger.info("card_id=%s not found in online store — using cold-start defaults", card_id)
        features = COLD_START_DEFAULTS
        feature_source = "cold_start_default"

    vector = build_feature_vector(features, model.feature_columns)
    fraud_score = model.predict_proba(vector)
    decision = decide(fraud_score)

    latency_ms = (time.perf_counter() - start) * 1000
    return ScoringResult(
        fraud_score=round(fraud_score, 4),
        decision=decision,
        feature_source=feature_source,
        latency_ms=round(latency_ms, 2),
    )