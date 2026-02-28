"""
main.py
-------
Slim FastAPI entrypoint — assembles middleware, routers, and lifespan.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.dependencies import lifespan
from app.routers import health, predict, report

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Respiratory Disease Classifier API",
    description=(
        "Upload a breath/cough WAV audio file and receive a respiratory "
        "condition prediction from a pre-trained Random Forest model. "
        "Generate AI-powered patient reports via Groq LLM."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS  (allow all origins — tighten for production)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(report.router)
