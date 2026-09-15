"""
Trains the XGBoost fraud classifier from the offline feature store.

Reads the same feature columns (`streaming.features.FEATURE_COLUMNS`) that
the streaming pipeline writes to Redis, so the model is trained on exactly
the feature set it will be served with — no separate "training feature
engineering" step that could drift from the online path.

If the offline store hasn't been populated yet (e.g. you haven't run
`make demo` / the streaming pipeline), this script falls back to generating
a synthetic-but-realistic dataset with the same schema, purely so the demo
works end-to-end out of the box. In a real deployment, delete
`generate_synthetic_fallback` and fail loudly instead.

Usage:
    python -m training.train
"""

from __future__ import annotations

import glob
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from streaming.features import FEATURE_COLUMNS, LABEL_COLUMN

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fraudlens.train")

OFFLINE_STORE_PATH = os.getenv("OFFLINE_STORE_PATH", "./data/offline_store")
MODEL_PATH = os.getenv("MODEL_PATH", "./training/model.pkl")
RANDOM_SEED = 42


def load_offline_features() -> pd.DataFrame:
    """Loads every partition written by streaming/sinks.py's OfflineSink."""
    parquet_files = glob.glob(f"{OFFLINE_STORE_PATH}/**/*.parquet", recursive=True)
    if not parquet_files:
        logger.warning(
            "No offline features found at %s — falling back to a synthetic "
            "dataset so the demo runs end-to-end. Run `make demo` first to "
            "train on real streamed features.",
            OFFLINE_STORE_PATH,
        )
        return generate_synthetic_fallback()

    df = pd.concat((pd.read_parquet(f) for f in parquet_files), ignore_index=True)
    logger.info("Loaded %d offline feature rows from %d partitions", len(df), len(parquet_files))
    return df


def generate_synthetic_fallback(n_rows: int = 20_000, fraud_rate: float = 0.03) -> pd.DataFrame:
    """Generates a dataset with the exact same schema the real offline store
    produces, with fraud rows exhibiting the same signal patterns injected by
    `producer/simulate_transactions.py` (higher z-score, higher velocity).
    Used only so `make demo` works before any real data has streamed through.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    # Distributions deliberately overlap (unlike a toy separable dataset) so
    # eval metrics reflect a realistic, imperfect fraud classifier rather than
    # a trivially separable synthetic signal.
    legit = pd.DataFrame(
        {
            "txn_count_1h": rng.poisson(2.5, n_legit),
            "txn_amount_sum_1h": rng.gamma(2, 45, n_legit),
            "txn_amount_avg_1h": rng.gamma(2, 22, n_legit),
            "txn_amount_max_1h": rng.gamma(2.5, 35, n_legit),
            "txn_count_24h": rng.poisson(9, n_legit),
            "txn_amount_avg_24h": rng.gamma(2, 26, n_legit),
            "distinct_merchant_categories_1h": rng.integers(1, 4, n_legit),
            "amount_zscore_vs_24h": rng.normal(0.3, 1.3, n_legit),
            LABEL_COLUMN: 0,
        }
    )

    fraud = pd.DataFrame(
        {
            "txn_count_1h": rng.poisson(6, n_fraud),
            "txn_amount_sum_1h": rng.gamma(2.2, 130, n_fraud),
            "txn_amount_avg_1h": rng.gamma(2.2, 75, n_fraud),
            "txn_amount_max_1h": rng.gamma(2.2, 140, n_fraud),
            "txn_count_24h": rng.poisson(12, n_fraud),
            "txn_amount_avg_24h": rng.gamma(2, 45, n_fraud),
            "distinct_merchant_categories_1h": rng.integers(1, 6, n_fraud),
            "amount_zscore_vs_24h": rng.normal(2.6, 1.8, n_fraud),
            LABEL_COLUMN: 1,
        }
    )

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED)
    return df.reset_index(drop=True)


def train_model(df: pd.DataFrame) -> tuple[xgb.XGBClassifier, dict]:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Offline dataset is missing expected feature columns: {missing}")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Fraud is rare — weight positives so the model doesn't just predict "legit" always.
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "pr_auc": average_precision_score(y_test, y_pred_proba),
    }
    logger.info("Eval metrics: %s", metrics)
    logger.info("\n%s", classification_report(y_test, y_pred, digits=3))

    return model, metrics


def save_model(model: xgb.XGBClassifier, path: str = MODEL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, f)
    logger.info("Model saved to %s", path)


def main() -> None:
    df = load_offline_features()
    model, metrics = train_model(df)
    save_model(model)
    logger.info("Training complete. PR-AUC=%.4f ROC-AUC=%.4f", metrics["pr_auc"], metrics["roc_auc"])


if __name__ == "__main__":
    main()