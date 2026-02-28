"""
app.dependencies
----------------
Application lifespan manager — loads model & pipeline at startup.
Exposes shared state via ``app.state``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, Request
from groq import AsyncGroq

from app.cache import PredictionCache
from app.config import get_settings
from model_utils import create_respiratory_pipeline

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic (replaces deprecated ``on_event``)."""
    settings = get_settings()

    # --- startup ---
    if not __import__("os").path.exists(settings.model_path):
        raise FileNotFoundError(
            f"Model file not found at '{settings.model_path}'. "
            "Make sure respiratory_classifier.pkl is in the project root."
        )

    app.state.model = joblib.load(settings.model_path)
    app.state.pipeline = create_respiratory_pipeline()
    app.state.pipeline.fit([])  # mark stateless transformers as fitted
    app.state.cache = PredictionCache(max_size=settings.cache_max_size)
    app.state.groq = AsyncGroq(api_key=settings.groq_api_key or None)
    app.state.settings = settings

    logger.info("✅  Model loaded from '%s'", settings.model_path)
    logger.info("✅  Classes: %s", list(app.state.model.classes_))

    yield  # app runs here

    # --- shutdown (cleanup if needed) ---
    logger.info("👋  Shutting down")


# ---------------------------------------------------------------------------
# Convenience helpers used by routers
# ---------------------------------------------------------------------------

def get_model(request: Request):
    return request.app.state.model


def get_pipeline(request: Request):
    return request.app.state.pipeline


def get_cache(request: Request) -> PredictionCache:
    return request.app.state.cache


def get_groq(request: Request) -> AsyncGroq:
    return request.app.state.groq
