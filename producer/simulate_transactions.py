"""
Generates a realistic stream of credit-card transactions — mostly legitimate,
with injected fraud patterns — and publishes them to the Kafka topic
`txn.events` for the streaming feature pipeline to consume.

Fraud is injected via three patterns real fraud systems actually look for:
  1. Card testing:        a burst of small, rapid-fire transactions on one card.
  2. Amount anomaly:      a transaction far outside the card's historical range.
  3. Geo/merchant jump:   an unfamiliar merchant category immediately after
                           a transaction in a completely different category.

Usage:
    python -m producer.simulate_transactions --count 500 --fraud-rate 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from dotenv import load_dotenv
from faker import Faker
from kafka import KafkaProducer

load_dotenv()

fake = Faker()

MERCHANT_CATEGORIES = [
    "grocery",
    "electronics_store",
    "gas_station",
    "restaurant",
    "online_retail",
    "pharmacy",
    "travel",
    "jewelry",
    "utilities",
    "entertainment",
]

# Categories fraudsters disproportionately target for cash-out / resale value.
HIGH_RISK_CATEGORIES = ["electronics_store", "jewelry", "travel"]


@dataclass
class Transaction:
    """Schema for a single transaction event on the `txn.events` topic.

    This shape is the contract between the producer and
    `streaming/feature_pipeline.py` — keep it in sync with the Spark schema
    defined there.
    """

    txn_id: str
    card_id: str
    amount: float
    merchant: str
    merchant_category: str
    timestamp: str
    is_fraud_label: int  # ground truth, for offline training/eval only —
    # NEVER exposed to the online scoring path.


class TransactionSimulator:
    """Generates a population of cards with individual spending baselines,
    then emits a mixed stream of normal activity and injected fraud."""

    def __init__(self, num_cards: int = 200, fraud_rate: float = 0.03) -> None:
        self.fraud_rate = fraud_rate
        self.cards = [f"card_{i:04d}" for i in range(num_cards)]
        # Each card has its own "normal" spending baseline so anomalies are
        # relative to the cardholder, not a global threshold.
        self.card_baselines = {
            card: {
                "avg_amount": round(random.uniform(15, 250), 2),
                "preferred_categories": random.sample(MERCHANT_CATEGORIES, k=3),
            }
            for card in self.cards
        }

    def _normal_transaction(self, card_id: str) -> Transaction:
        baseline = self.card_baselines[card_id]
        amount = max(1.0, round(random.gauss(baseline["avg_amount"], baseline["avg_amount"] * 0.3), 2))
        category = random.choice(baseline["preferred_categories"])
        return Transaction(
            txn_id=str(uuid.uuid4()),
            card_id=card_id,
            amount=amount,
            merchant=fake.company(),
            merchant_category=category,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_fraud_label=0,
        )

    def _fraud_transaction(self, card_id: str) -> Transaction:
        baseline = self.card_baselines[card_id]
        pattern = random.choice(["card_testing", "amount_anomaly", "category_jump"])

        if pattern == "card_testing":
            amount = round(random.uniform(1.0, 5.0), 2)  # small "is this card alive" charges
            category = random.choice(MERCHANT_CATEGORIES)
        elif pattern == "amount_anomaly":
            amount = round(baseline["avg_amount"] * random.uniform(8, 25), 2)
            category = random.choice(HIGH_RISK_CATEGORIES)
        else:  # category_jump
            amount = round(random.uniform(200, 2000), 2)
            unfamiliar = [c for c in MERCHANT_CATEGORIES if c not in baseline["preferred_categories"]]
            category = random.choice(unfamiliar or MERCHANT_CATEGORIES)

        return Transaction(
            txn_id=str(uuid.uuid4()),
            card_id=card_id,
            amount=amount,
            merchant=fake.company(),
            merchant_category=category,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_fraud_label=1,
        )

    def next_transaction(self) -> Transaction:
        card_id = random.choice(self.cards)
        if random.random() < self.fraud_rate:
            return self._fraud_transaction(card_id)
        return self._normal_transaction(card_id)


def build_producer(bootstrap_servers: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        acks="all",
        retries=5,
        linger_ms=5,
    )


def run(count: int, fraud_rate: float, rate_per_sec: float, bootstrap_servers: str, topic: str) -> None:
    simulator = TransactionSimulator(fraud_rate=fraud_rate)
    producer = build_producer(bootstrap_servers)
    delay = 1.0 / rate_per_sec if rate_per_sec > 0 else 0

    sent, flagged = 0, 0
    try:
        for _ in range(count):
            txn = simulator.next_transaction()
            producer.send(topic, key=txn.card_id, value=asdict(txn))
            sent += 1
            flagged += txn.is_fraud_label
            if sent % 50 == 0:
                print(f"[producer] sent={sent} injected_fraud={flagged}")
            if delay:
                time.sleep(delay)
    finally:
        producer.flush()
        producer.close()
        print(f"[producer] done. total={sent} injected_fraud={flagged} ({flagged / max(sent,1):.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a transaction stream into Kafka.")
    parser.add_argument("--count", type=int, default=500, help="number of transactions to emit")
    parser.add_argument("--fraud-rate", type=float, default=0.03, help="fraction of injected fraud, e.g. 0.03")
    parser.add_argument("--rate-per-sec", type=float, default=10.0, help="transactions per second (0 = as fast as possible)")
    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"),
    )
    parser.add_argument("--topic", default=os.getenv("KAFKA_TOPIC_TRANSACTIONS", "txn.events"))
    args = parser.parse_args()

    run(
        count=args.count,
        fraud_rate=args.fraud_rate,
        rate_per_sec=args.rate_per_sec,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
    )


if __name__ == "__main__":
    main()