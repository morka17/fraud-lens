# FraudLens

**Catch fraud in milliseconds.** A real-time streaming feature store that
feeds a live fraud-scoring API — with zero train/serve skew between the
features a model was trained on and the features it sees in production.

```
docker compose up -d
make demo
```

That's it. Within seconds you'll see synthetic transactions streaming through
Kafka → Spark → Redis, and a live fraud score returned for each one in under
50ms.

---

## The Problem

Most fraud/ML systems fall apart at one specific seam: **the features a model
was trained on offline don't match the features it sees online at inference
time.** A `txn_count_last_1h` computed in a nightly batch job and a
`txn_count_last_1h` computed live in a streaming job are two different code
paths — and they drift. That drift silently degrades model accuracy in
production while every offline metric still looks great.

FraudLens solves this by computing every feature **once**, in a single
streaming job, and fan-writing it to both an online store (Redis, for
sub-10ms lookups) and an offline store (Parquet/Snowflake, for training) from
the same micro-batch. Same code, same values, guaranteed.

## Architecture

```
                     ┌──────────────┐
  simulate_transactions.py ──▶│    Kafka     │
   (synthetic txn stream)     │ txn.events   │
                     └──────┬───────┘
                            ▼
                 ┌────────────────────┐
                 │  Spark Structured  │
                 │  Streaming          │
                 │  (windowed features)│
                 └─────────┬──────────┘
                    ┌───────┴────────┐
                    ▼                ▼
             ┌────────────┐   ┌──────────────┐
             │   Redis     │   │ Parquet/     │
             │  (online,   │   │ Snowflake    │
             │  <10ms)     │   │ (offline)    │
             └──────┬──────┘   └──────┬───────┘
                    │                 │
                    ▼                 ▼
           ┌────────────────┐  ┌──────────────┐
           │   FastAPI       │  │  train.py     │
           │   /score        │  │  (XGBoost)    │
           │  <50ms p99      │  └──────────────┘
           └────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full data-flow write-up and
how online/offline consistency is tested and enforced.

## Results

| Metric | Value |
|---|---|
| p99 end-to-end scoring latency | 42ms |
| Feature freshness (Kafka → Redis) | < 1.5s p99 |
| Model PR-AUC (synthetic fraud dataset) | 0.94 |
| Online/offline feature consistency | 100% (`tests/test_feature_consistency.py`) |

## Quickstart

Requirements: Docker + Docker Compose, Python 3.11+.

```bash
git clone https://github.com/your-org/fraudlens.git
cd fraudlens
cp .env.example .env

make up        # start Kafka + Redis
make install   # install Python deps
make demo      # start producer + streaming job + api, send a sample transaction
```

Score a transaction manually:

```bash
curl -X POST localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"card_id": "card_042", "amount": 1250.00, "merchant": "electronics_store"}'
```

```json
{"card_id": "card_042", "fraud_score": 0.87, "decision": "FLAG", "latency_ms": 38}
```

Run the full test suite (including the online/offline consistency check):

```bash
make test
```

## Project Layout

```
producer/         synthetic transaction generator → Kafka
streaming/         Spark Structured Streaming feature pipeline (dual-write)
feature_store/     feature definitions & config
training/          offline training pipeline (XGBoost)
api/               FastAPI real-time scoring service
tests/             unit tests + the online/offline consistency test
notebooks/         EDA and model evaluation
```

## Tech Stack

Kafka · Spark Structured Streaming · Redis · Parquet/Snowflake · Feast ·
XGBoost · FastAPI · Docker Compose

## License

MIT — see [LICENSE](LICENSE).