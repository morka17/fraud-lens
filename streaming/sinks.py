"""
Dual-write sink used by the Spark `foreachBatch` call in feature_pipeline.py.

The entire online/offline consistency guarantee lives in this one function:
`write_batch` takes a single already-computed micro-batch DataFrame and
writes it to Redis and to the offline store WITHOUT recomputing anything.
There is exactly one code path that produces feature values; this module
only fans the same values out to two destinations.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import redis
from pyspark.sql import DataFrame

from streaming.features import FEATURE_COLUMNS, redis_feature_key

logger = logging.getLogger("fraudlens.sinks")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_FEATURE_TTL_SECONDS = int(os.getenv("REDIS_FEATURE_TTL_SECONDS", "86400"))

OFFLINE_STORE_PATH = os.getenv("OFFLINE_STORE_PATH", "./data/offline_store")
USE_SNOWFLAKE = os.getenv("SNOWFLAKE_ACCOUNT") is not None


class RedisOnlineSink:
    """Writes the latest feature vector per card_id to Redis for sub-10ms
    reads by the FastAPI scoring service."""

    def __init__(self) -> None:
        self._client: redis.Redis | None = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
            )
        return self._client

    def write(self, rows: list[dict]) -> int:
        """Pipeline-writes a batch of feature rows. Returns the number of
        keys written."""
        if not rows:
            return 0
        pipe = self.client.pipeline(transaction=False)
        for row in rows:
            key = redis_feature_key(row["card_id"])
            feature_payload = {col: row.get(col) for col in FEATURE_COLUMNS}
            feature_payload["_updated_at"] = datetime.now(timezone.utc).isoformat()
            pipe.set(key, json.dumps(feature_payload), ex=REDIS_FEATURE_TTL_SECONDS)
        results = pipe.execute()
        return len(results)


class OfflineSink:
    """Appends the same feature rows to the offline store used for training.

    Defaults to local Parquet (partitioned by event_date) for the demo.
    Swap `write` for a Snowflake `COPY INTO` / connector call in production —
    the interface and the values passed in stay identical either way.
    """

    def __init__(self, path: str = OFFLINE_STORE_PATH) -> None:
        self.path = path

    def write(self, df: DataFrame) -> None:
        if USE_SNOWFLAKE:
            self._write_snowflake(df)
        else:
            self._write_parquet(df)

    def _write_parquet(self, df: DataFrame) -> None:
        (
            df.withColumn("event_date", df["window_end"].cast("date"))
            .write.mode("append")
            .partitionBy("event_date")
            .parquet(self.path)
        )

    def _write_snowflake(self, df: DataFrame) -> None:  # pragma: no cover - requires live warehouse
        options = {
            "sfURL": f"{os.environ['SNOWFLAKE_ACCOUNT']}.snowflakecomputing.com",
            "sfUser": os.environ["SNOWFLAKE_USER"],
            "sfPassword": os.environ["SNOWFLAKE_PASSWORD"],
            "sfWarehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
            "sfDatabase": os.environ["SNOWFLAKE_DATABASE"],
            "sfSchema": os.environ["SNOWFLAKE_SCHEMA"],
            "dbtable": "FRAUD_FEATURES_OFFLINE",
        }
        df.write.format("net.snowflake.spark.snowflake").options(**options).mode("append").save()


_redis_sink = RedisOnlineSink()
_offline_sink = OfflineSink()


def write_batch(batch_df: DataFrame, batch_id: int) -> None:
    """The `foreachBatch` entry point wired up in feature_pipeline.py.

    Both writes below read from the SAME `batch_df` produced by a single
    Spark micro-batch — this is the whole ballgame for consistency. If you
    ever find yourself computing a feature differently for one sink than the
    other, you have broken the core guarantee of this project.
    """
    batch_df.persist()
    try:
        row_count = batch_df.count()
        logger.info("batch_id=%s rows=%s writing to Redis + offline store", batch_id, row_count)

        # 1. Online store — collect is safe here because this is a per-card
        #    aggregate batch (small), not raw event volume.
        rows = [r.asDict() for r in batch_df.collect()]
        written = _redis_sink.write(rows)
        logger.info("batch_id=%s redis_keys_written=%s", batch_id, written)

        # 2. Offline store — same DataFrame, no recomputation.
        _offline_sink.write(batch_df)
        logger.info("batch_id=%s offline_write_complete", batch_id)
    finally:
        batch_df.unpersist()