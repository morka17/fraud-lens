"""
FastAPI application for FraudLens real-time scoring.

    uvicorn api.main:app --reload

The model and Redis connection are initialized once at startup (not per
request) so the per-request path is just: Redis lookup → vector → predict.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse, ScoreRequest, ScoreResponse
from api.scoring import FeatureClient, ModelWrapper, score_transaction

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fraudlens.api")

# Module-level singletons, populated on startup. Kept as plain globals rather
# than FastAPI Depends() factories to avoid re-instantiating a Redis
# connection pool on every request.
feature_client: FeatureClient | None = None
model: ModelWrapper | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global feature_client, model
    feature_client = FeatureClient()
    model = ModelWrapper()
    try:
        model.load()
    except FileNotFoundError as e:
        # Don't crash the process — /health will report model_loaded=False
        # and /score will 503 until `make train` has been run. This makes
        # the failure mode legible instead of a boot-loop.
        logger.warning("Model not loaded at startup: %s", e)
    yield


app = FastAPI(
    title="FraudLens",
    description="Real-time fraud scoring API backed by a streaming feature store.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    redis_ok = feature_client.ping() if feature_client else False
    model_ok = model.is_loaded if model else False
    return HealthResponse(
        status="ok" if (redis_ok and model_ok) else "degraded",
        redis_connected=redis_ok,
        model_loaded=model_ok,
    )


@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score(request: ScoreRequest) -> ScoreResponse:
    if model is None or not model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Run `python -m training.train` and restart the API.",
        )
    if feature_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feature store client is not initialized.",
        )

    result = score_transaction(
        card_id=request.card_id,
        feature_client=feature_client,
        model=model,
    )

    return ScoreResponse(
        card_id=request.card_id,
        fraud_score=result.fraud_score,
        decision=result.decision,
        latency_ms=result.latency_ms,
        feature_source=result.feature_source,
    )