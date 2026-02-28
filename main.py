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
from app.routers import drugs, health, heart, lab, predict, report, scan, symptoms

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Respiratory Disease Classifier API",
    description=(
        "Upload a breath/cough WAV audio file and receive a respiratory "
        "condition prediction from a pre-trained Random Forest model. "
        "Generate AI-powered patient reports via Groq LLM. "
        "Assess heart disease risk with multi-step AI analysis."
    ),
    version="2.1.0",
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
app.include_router(heart.router)
app.include_router(scan.router)
app.include_router(symptoms.router)
app.include_router(drugs.router)
app.include_router(lab.router)
