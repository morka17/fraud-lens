"""
Spark Structured Streaming job: consumes raw transaction events from Kafka,
computes windowed features (defined once in `streaming/features.py`), and
dual-writes each micro-batch to Redis (online) and the offline store
(`streaming/sinks.py`).

Run standalone:
    python -m streaming.feature_pipeline

Or submit via spark-submit in a real cluster:
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
        streaming/feature_pipeline.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from streaming.features import (
    ENTITY_KEY,
    amount_zscore_expr,
    build_24h_aggregations,
    build_windowed_aggregations,
)
from streaming.sinks import write_batch

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "txn.events")
CHECKPOINT_DIR = os.getenv("SPARK_CHECKPOINT_DIR", "./streaming/checkpoints")
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

# Must match the `Transaction` dataclass in producer/simulate_transactions.py
TRANSACTION_SCHEMA = StructType(
    [
        StructField("txn_id", StringType(), nullable=False),
        StructField("card_id", StringType(), nullable=False),
        StructField("amount", DoubleType(), nullable=False),
        StructField("merchant", StringType(), nullable=False),
        StructField("merchant_category", StringType(), nullable=False),
        StructField("timestamp", StringType(), nullable=False),
        StructField("is_fraud_label", IntegerType(), nullable=True),
    ]
)


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("fraudlens-feature-pipeline")
        .master(SPARK_MASTER)
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
        )
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def read_transaction_stream(spark: SparkSession):
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = (
        raw.selectExpr("CAST(value AS STRING) AS json_value", "timestamp AS kafka_timestamp")
        .select(F.from_json("json_value", TRANSACTION_SCHEMA).alias("txn"), "kafka_timestamp")
        .select("txn.*", "kafka_timestamp")
        .withColumn("event_time", F.to_timestamp("timestamp"))
    )
    return parsed


def compute_windowed_features(parsed_df):
    """Computes 1h and 24h windowed aggregations per card, joins them, and
    derives the amount z-score. All aggregation logic is delegated to
    `streaming/features.py` so this function never defines a feature inline.
    """
    watermarked = parsed_df.withWatermark("event_time", "5 minutes")

    agg_1h = (
        watermarked.groupBy(
            F.col(ENTITY_KEY),
            F.window("event_time", "1 hour").alias("window_1h"),
        )
        .agg(*build_windowed_aggregations())
        .withColumnRenamed("window_1h", "window")
    )

    agg_24h = (
        watermarked.groupBy(
            F.col(ENTITY_KEY),
            F.window("event_time", "24 hours").alias("window_24h"),
        )
        .agg(*build_24h_aggregations())
    )

    joined = agg_1h.join(
        agg_24h,
        on=[
            agg_1h[ENTITY_KEY] == agg_24h[ENTITY_KEY],
            agg_1h["window"]["start"] >= agg_24h["window_24h"]["start"],
            agg_1h["window"]["end"] <= agg_24h["window_24h"]["end"],
        ],
        how="left",
    ).select(
        agg_1h[ENTITY_KEY],
        agg_1h["window"]["start"].alias("window_start"),
        agg_1h["window"]["end"].alias("window_end"),
        "txn_count_1h",
        "txn_amount_sum_1h",
        "txn_amount_avg_1h",
        "txn_amount_max_1h",
        "distinct_merchant_categories_1h",
        "txn_count_24h",
        "txn_amount_avg_24h",
        "txn_amount_stddev_24h",
    )

    with_zscore = joined.withColumn(
        "amount_zscore_vs_24h",
        amount_zscore_expr("txn_amount_max_1h", "txn_amount_avg_24h", "txn_amount_stddev_24h"),
    )

    return with_zscore


def run() -> None:
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    parsed = read_transaction_stream(spark)
    features = compute_windowed_features(parsed)

    query = (
        features.writeStream.outputMode("update")
        .foreachBatch(write_batch)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .trigger(processingTime="5 seconds")
        .start()
    )

    print(f"[feature_pipeline] streaming started — checkpoint: {CHECKPOINT_DIR}")
    query.awaitTermination()


if __name__ == "__main__":
    run()