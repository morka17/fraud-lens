"""Tests for the FastAPI scoring service (api/main.py).

Redis and the model are faked/stubbed here — these are API contract and
routing tests, not infrastructure tests. The online/offline data-correctness
guarantee is covered separately and much more rigorously by
tests/test_feature_consistency.py.
"""

from __future__ import annotations

import json

import fakeredis
import pytest
from fastapi.testclient import TestClient

import api.main as main_module
from api.scoring import FeatureClient, ModelWrapper


class _StubModel:
    """Deterministic stand-in for the real XGBoost model so tests don't
    depend on a trained artifact existing on disk."""

    def predict_proba(self, X):
        # High amount_zscore (last feature) => high fraud score; otherwise low.
        vector = X[0]
        zscore = vector[-1]
        score = 0.95 if zscore > 2.5 else 0.05
        return [[1 - score, score]]


@pytest.fixture()
def client(monkeypatch):
    fake_redis = fakeredis.FakeRedis(decode_responses=True)

    feature_client = FeatureClient(client=fake_redis)

    model = ModelWrapper()
    model.model = _StubModel()  # bypass .load(), no file needed

    monkeypatch.setattr(main_module, "feature_client", feature_client)
    monkeypatch.setattr(main_module, "model", model)

    return TestClient(main_module.app), fake_redis


def test_health_reports_ok_when_dependencies_ready(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["redis_connected"] is True
    assert body["model_loaded"] is True


def test_score_uses_online_features_when_present(client):
    test_client, fake_redis = client
    fake_redis.set(
        "features:card_0001",
        json.dumps(
            {
                "txn_count_1h": 2,
                "txn_amount_sum_1h": 80.0,
                "txn_amount_avg_1h": 40.0,
                "txn_amount_max_1h": 60.0,
                "txn_count_24h": 6,
                "txn_amount_avg_24h": 45.0,
                "distinct_merchant_categories_1h": 1,
                "amount_zscore_vs_24h": 0.4,
            }
        ),
    )

    response = test_client.post(
        "/score",
        json={"card_id": "card_0001", "amount": 42.0, "merchant": "Corner Store", "merchant_category": "grocery"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["card_id"] == "card_0001"
    assert body["feature_source"] == "online_store"
    assert body["decision"] == "ALLOW"
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert body["latency_ms"] >= 0


def test_score_flags_high_zscore_transaction(client):
    test_client, fake_redis = client
    fake_redis.set(
        "features:card_0002",
        json.dumps(
            {
                "txn_count_1h": 9,
                "txn_amount_sum_1h": 2400.0,
                "txn_amount_avg_1h": 270.0,
                "txn_amount_max_1h": 980.0,
                "txn_count_24h": 14,
                "txn_amount_avg_24h": 310.0,
                "distinct_merchant_categories_1h": 5,
                "amount_zscore_vs_24h": 3.9,
            }
        ),
    )

    response = test_client.post(
        "/score",
        json={"card_id": "card_0002", "amount": 980.0, "merchant": "Electronics Depot", "merchant_category": "electronics_store"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in {"FLAG", "BLOCK"}
    assert body["fraud_score"] > 0.5


def test_score_falls_back_to_cold_start_for_unknown_card(client):
    test_client, _ = client
    response = test_client.post(
        "/score",
        json={"card_id": "card_never_seen", "amount": 25.0, "merchant": "Cafe", "merchant_category": "restaurant"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["feature_source"] == "cold_start_default"


def test_score_rejects_invalid_payload(client):
    test_client, _ = client
    response = test_client.post("/score", json={"card_id": "card_0001", "amount": -5.0, "merchant": "x"})
    assert response.status_code == 422


def test_score_returns_503_when_model_not_loaded(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(main_module, "model", None)

    response = test_client.post(
        "/score",
        json={"card_id": "card_0001", "amount": 10.0, "merchant": "x", "merchant_category": "grocery"},
    )
    assert response.status_code == 503