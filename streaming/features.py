"""
Single source of truth for every feature FraudLens computes.

This module is imported by BOTH `streaming/feature_pipeline.py` (which
computes features live from the Kafka stream) and `training/train.py` (which
reads the already-materialized offline features for model training). Neither
consumer re-implements a feature's logic — they only read the names and
schema defined here. This is what makes online/offline consistency a
structural property of the codebase rather than a hope.

If you add a feature, add it here first, then wire it into the Spark
aggregation in `feature_pipeline.py`. Do not compute a feature ad-hoc in any
other file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import Column

# NOTE: pyspark is imported lazily inside the functions below, not at module
# scope. FEATURE_COLUMNS/LABEL_COLUMN/redis_feature_key() are plain Python
# and are imported directly by training/train.py and api/scoring.py, which
# have no reason to depend on a Spark installation. Only the streaming
# pipeline (which already requires pyspark) calls the aggregation builders.

# ---------------------------------------------------------------------------
# Feature schema — the contract every downstream consumer relies on.
# ---------------------------------------------------------------------------

ENTITY_KEY = "card_id"

# Every feature name that must exist, byte-for-byte identical, in both the
# Redis online store and the offline Parquet/Snowflake table.
FEATURE_COLUMNS: list[str] = [
    "txn_count_1h",
    "txn_amount_sum_1h",
    "txn_amount_avg_1h",
    "txn_amount_max_1h",
    "txn_count_24h",
    "txn_amount_avg_24h",
    "distinct_merchant_categories_1h",
    "amount_zscore_vs_24h",
]

LABEL_COLUMN = "is_fraud_label"


@dataclass(frozen=True)
class WindowSpec:
    """Declarative definition of a single windowed aggregation."""

    name: str
    duration: str  # Spark time-window string, e.g. "1 hour"
    watermark: str  # allowed lateness, e.g. "2 minutes"


WINDOWS = {
    "1h": WindowSpec(name="1h", duration="1 hour", watermark="2 minutes"),
    "24h": WindowSpec(name="24h", duration="24 hours", watermark="5 minutes"),
}


def build_windowed_aggregations(event_time_col: str = "timestamp") -> list["Column"]:
    """Returns the list of Spark aggregation expressions used to compute every
    feature in FEATURE_COLUMNS from raw transaction rows grouped by
    (card_id, window).

    Used identically whether the pipeline is running in streaming mode
    (feature_pipeline.py) or would ever need to be replayed in batch mode
    for a backfill — same function, same result.
    """
    from pyspark.sql import functions as F

    return [
        F.count("*").alias("txn_count_1h"),
        F.sum("amount").alias("txn_amount_sum_1h"),
        F.avg("amount").alias("txn_amount_avg_1h"),
        F.max("amount").alias("txn_amount_max_1h"),
        F.approx_count_distinct("merchant_category").alias("distinct_merchant_categories_1h"),
    ]


def build_24h_aggregations() -> list["Column"]:
    """24-hour window aggregations, computed separately since they use a
    wider window than the 1h feature group."""
    from pyspark.sql import functions as F

    return [
        F.count("*").alias("txn_count_24h"),
        F.avg("amount").alias("txn_amount_avg_24h"),
        F.stddev("amount").alias("txn_amount_stddev_24h"),
    ]


def amount_zscore_expr(amount_col: str, avg_24h_col: str, stddev_24h_col: str) -> "Column":
    """z-score of the current transaction amount relative to the card's
    trailing 24h distribution — the single strongest anomaly signal in the
    feature set. Guards against divide-by-zero for cards with too little
    history (stddev is null/0 for a brand-new or single-transaction card).
    """
    from pyspark.sql import functions as F

    stddev_safe = F.when(F.col(stddev_24h_col).isNull() | (F.col(stddev_24h_col) == 0), F.lit(1.0)).otherwise(
        F.col(stddev_24h_col)
    )
    return ((F.col(amount_col) - F.col(avg_24h_col)) / stddev_safe).alias("amount_zscore_vs_24h")


def redis_feature_key(card_id: str) -> str:
    """Canonical Redis key format for a card's feature vector. Used by both
    the streaming sink (write) and the API feature client (read) — defined
    once here so the two never disagree on key format.
    """
    return f"features:{card_id}"


def offline_partition_columns() -> list[str]:
    """Partition scheme for the offline Parquet/Snowflake table."""
    return ["event_date"]