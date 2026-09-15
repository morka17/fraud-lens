.DEFAULT_GOAL := help
.PHONY: help up down install produce stream api demo test test-consistency lint clean logs

COMPOSE := docker compose

help: ## Show this help
	@echo "FraudLens — real-time fraud detection engine"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

up: ## Start Kafka, Redis, and supporting infra
	$(COMPOSE) up -d
	@echo "Waiting for Kafka + Redis to be healthy..."
	@sleep 5
	@echo "✓ Infra is up. Run 'make demo' next."

down: ## Stop and remove all containers
	$(COMPOSE) down -v

install: ## Install Python dependencies
	pip install --upgrade pip
	pip install -e ".[dev]"

produce: ## Start the synthetic transaction producer
	python -m producer.simulate_transactions

stream: ## Submit the Spark Structured Streaming feature pipeline
	python -m streaming.feature_pipeline

api: ## Run the FastAPI scoring service
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

train: ## Train the XGBoost model from the offline feature store
	python -m training.train

demo: ## One-command end-to-end demo: producer + streaming + api + sample request
	@echo "▶ Starting streaming pipeline in the background..."
	@python -m streaming.feature_pipeline & echo $$! > .stream.pid
	@sleep 3
	@echo "▶ Starting transaction producer in the background..."
	@python -m producer.simulate_transactions --count 200 & echo $$! > .producer.pid
	@sleep 3
	@echo "▶ Starting API..."
	@uvicorn api.main:app --host 0.0.0.0 --port 8000 & echo $$! > .api.pid
	@sleep 3
	@echo "▶ Sending a sample scoring request:"
	@curl -s -X POST localhost:8000/score \
		-H "Content-Type: application/json" \
		-d '{"card_id": "card_042", "amount": 1250.00, "merchant": "electronics_store"}' | python -m json.tool
	@echo ""
	@echo "Demo running. API at http://localhost:8000/docs — Ctrl+C then 'make stop-demo' to clean up."

stop-demo: ## Stop background processes started by `make demo`
	-kill $$(cat .stream.pid) 2>/dev/null; rm -f .stream.pid
	-kill $$(cat .producer.pid) 2>/dev/null; rm -f .producer.pid
	-kill $$(cat .api.pid) 2>/dev/null; rm -f .api.pid

test: ## Run the full test suite
	pytest tests/ -v

test-consistency: ## Run only the online/offline feature consistency test
	pytest tests/test_feature_consistency.py -v

lint: ## Run formatting and static checks
	ruff check .
	black --check .
	mypy .

logs: ## Tail logs from all infra containers
	$(COMPOSE) logs -f

clean: ## Remove caches, checkpoints, and build artifacts
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__ \
		streaming/checkpoints .stream.pid .producer.pid .api.pid
	find . -name "*.pyc" -delete