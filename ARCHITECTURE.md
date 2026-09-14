# Architecture

## Data Flow

```
producer/simulate_transactions.py
        │  Avro/JSON transaction events
        ▼
     Kafka (topic: txn.events, partitioned by card_id)
        │
        ▼
streaming/feature_pipeline.py   (Spark Structured Streaming)
        │  - parses & validates events
        │  - computes windowed features (velocity, rolling sum/count,
        │    merchant risk aggregates) — defined once in streaming/features.py
        │
        │  foreachBatch, same micro-batch, same computed values:
        ├────────────────────────────┬───────────────────────────┐
        ▼                            ▼
   Redis (online store)      Parquet / Snowflake (offline store)
   key: features:{card_id}   partitioned by event_date
   TTL: 24h                  append-only, immutable
        │                            │
        ▼                            ▼
   api/scoring.py               training/train.py
   (live inference,             (point-in-time correct
    <10ms lookup)                training set, XGBoost)
        │
        ▼
   FastAPI /score endpoint
   feature vector → XGBoost → fraud_score
```

## The One Hard Problem: Online/Offline Consistency

Every real-time ML system with a feature store hits the same failure mode:
the feature value used at training time and the feature value used at
inference time are computed by **two separate pieces of code** (a batch job
and a streaming job), and over time those two implementations drift apart.
The model then trains on values it will never actually see in production —
this is train/serve skew, and it's usually invisible until fraud starts
slipping through and nobody can explain why.

**How FraudLens avoids it:**

1. **One feature definition, one execution.** `streaming/features.py` is the
   single source of truth for every feature transformation (e.g.
   `txn_count_1h`, `avg_amount_7d`). There is no separate batch
   re-implementation of these functions anywhere else in the codebase.

2. **One micro-batch, two writes.** `streaming/sinks.py` takes the *already
   computed* feature DataFrame from a single Spark micro-batch and writes it
   to Redis and to the offline Parquet/Snowflake table in the same
   `foreachBatch` call. The values that land in both stores are byte-for-byte
   the same — there is no second computation path that could drift.

3. **Point-in-time correctness for training.** `training/train.py` builds the
   training set by joining labels against the offline store as of the label's
   timestamp, never against the current (mutable) online state, so the model
   never trains on information that wasn't actually available at that moment.

4. **A test that proves it.** `tests/test_feature_consistency.py` replays a
   batch of synthetic transactions through the pipeline and asserts that the
   feature vector read from Redis for a given `card_id` at time `t` exactly
   matches the row written to the offline store for the same `card_id` and
   window. This test is the load-bearing proof of the whole design — if it's
   green, train/serve skew for these features is structurally impossible, not
   just "unlikely."

## Latency Budget (p99 target: 50ms)

| Hop | Budget |
|---|---|
| Redis feature lookup | 10ms |
| Feature vector assembly | 2ms |
| XGBoost inference | 8ms |
| Network + serialization | 10ms |
| Headroom | 20ms |

## Why Not Just Use Feast for Everything?

Feast is used only for feature *definitions and registry* (`feature_store/`)
— the dual-write mechanics live in plain Spark/Redis/Parquet code in
`streaming/` so the core consistency guarantee is easy to read, audit, and
test without needing to trust a framework's internals.