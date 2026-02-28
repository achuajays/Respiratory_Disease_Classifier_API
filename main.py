"""
main.py
-------
FastAPI server for respiratory disease classification.
Upload a WAV file → receive a prediction + confidence scores.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Docs:
    http://localhost:8000/docs
"""

import hashlib
import os
import tempfile
from enum import Enum
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # reads .env file in project root

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import AsyncGroq
from pydantic import BaseModel, Field

from model_utils import create_respiratory_pipeline


# ---------------------------------------------------------------------------
# Enums & request models
# ---------------------------------------------------------------------------

class RespiratoryDisease(str, Enum):
    asthma = "Asthma"
    bronchiectasis = "Bronchiectasis"
    bronchiolitis = "Bronchiolitis"
    copd = "COPD"
    healthy = "Healthy"
    lrti = "LRTI"
    pneumonia = "Pneumonia"
    urti = "URTI"


class ReportRequest(BaseModel):
    disease: RespiratoryDisease = Field(..., description="Diagnosed respiratory condition")
    age: Optional[int] = Field(None, ge=0, le=150, description="Patient age in years")
    height: Optional[float] = Field(None, gt=0, description="Patient height in cm")
    weight: Optional[float] = Field(None, gt=0, description="Patient weight in kg")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Respiratory Disease Classifier API",
    description=(
        "Upload a breath/cough WAV audio file and receive a respiratory "
        "condition prediction from a pre-trained Random Forest model."
    ),
    version="1.0.0",
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
# Globals  (populated on startup)
# ---------------------------------------------------------------------------

model = None
preprocessing_pipeline = None

# In-memory prediction cache (keyed by SHA-256 of file content)
# Maxsize=128 keeps memory bounded; same file won't re-run the pipeline.
PREDICTION_CACHE: dict[str, dict] = {}
CACHE_MAX_SIZE = 128

MODEL_PATH = os.path.join(os.path.dirname(__file__), "respiratory_classifier.pkl")

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


@app.on_event("startup")
def load_model():
    """Load model and preprocessing pipeline once at startup (CPU-only)."""
    global model, preprocessing_pipeline

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Make sure respiratory_classifier.pkl is in the project root."
        )

    model = joblib.load(MODEL_PATH)
    preprocessing_pipeline = create_respiratory_pipeline()
    # sklearn Pipeline.transform() requires fit() to have been called first,
    # even for stateless transformers.  fit([]) is safe — every transformer
    # simply returns self without touching the data.
    preprocessing_pipeline.fit([])
    print(f"✅  Model loaded from '{MODEL_PATH}'")
    print(f"✅  Classes: {list(model.classes_)}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", summary="Health check")
def root():
    """Simple health-check endpoint."""
    return {"status": "ok", "message": "Respiratory Classifier API is running 🫁"}


@app.get("/classes", summary="List model classes")
def list_classes():
    """Return the respiratory condition labels the model can predict."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"classes": list(model.classes_)}


@app.post("/predict", summary="Classify a respiratory audio sample")
async def predict(file: UploadFile = File(..., description="WAV audio file to classify")):
    """
    Upload a WAV file and receive:

    - **prediction** – most likely respiratory condition
    - **confidence** – probability of the predicted class (0–1)
    - **all_probabilities** – probability for every class
    """
    if model is None or preprocessing_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not ready. Try again shortly.")

    # Validate content type loosely (accept wav / octet-stream)
    filename = file.filename or ""
    if not (
        filename.lower().endswith(".wav")
        or (file.content_type or "").startswith("audio/")
        or file.content_type == "application/octet-stream"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only WAV audio files are supported. Please upload a .wav file.",
        )

    # ------------------------------------------------------------------
    # Save upload to a temporary file so librosa can read it from disk
    # ------------------------------------------------------------------
    temp_path: str | None = None
    try:
        content = await file.read()

        # -----------------------------------------------------------------
        # Cache check  (hash the raw bytes → skip pipeline if seen before)
        # -----------------------------------------------------------------
        file_hash = hashlib.sha256(content).hexdigest()

        if file_hash in PREDICTION_CACHE:
            return JSONResponse(content=PREDICTION_CACHE[file_hash])

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav", prefix="resp_"
        ) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        # -----------------------------------------------------------------
        # Feature extraction  (AudioLoader → Trimmer → Extractor → Stats)
        # -----------------------------------------------------------------
        features_df = preprocessing_pipeline.transform([temp_path])

        # -----------------------------------------------------------------
        # Inference
        # -----------------------------------------------------------------
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        confidence = float(max(probabilities))
        all_probs = {
            cls: round(float(prob), 6)
            for cls, prob in zip(model.classes_, probabilities)
        }

        result = {
            "prediction": str(prediction),
            "confidence": round(confidence, 6),
            "all_probabilities": all_probs,
        }

        # Store in cache (evict oldest if full)
        if len(PREDICTION_CACHE) >= CACHE_MAX_SIZE:
            oldest_key = next(iter(PREDICTION_CACHE))
            del PREDICTION_CACHE[oldest_key]
        PREDICTION_CACHE[file_hash] = result

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
    finally:
        # Always clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------------------------
# Groq client (async, uses GROQ_API_KEY env var)
# ---------------------------------------------------------------------------

groq_client = AsyncGroq()

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


@app.post("/report", summary="Generate a patient report using AI")
async def generate_report(req: ReportRequest):
    """
    Generate a comprehensive patient report for a respiratory condition
    using Groq's Llama 4 Scout model.

    - **disease** (required) – one of the 8 classified conditions
    - **age**, **height**, **weight** – optional patient details
    """

    # ----- Build patient context -----
    patient_info_parts = [f"Diagnosed condition: **{req.disease.value}**"]
    if req.age is not None:
        patient_info_parts.append(f"Age: {req.age} years")
    if req.height is not None:
        patient_info_parts.append(f"Height: {req.height} cm")
    if req.weight is not None:
        patient_info_parts.append(f"Weight: {req.weight} kg")
    if req.height and req.weight:
        bmi = round(req.weight / ((req.height / 100) ** 2), 1)
        patient_info_parts.append(f"BMI: {bmi}")

    patient_context = "\n".join(patient_info_parts)

    # ----- System prompt -----
    system_prompt = """You are a senior pulmonologist AI assistant.
Generate a detailed, professional patient report based on the information provided.

The report must include ALL of the following sections:

1. **Patient Summary** – Demographics and key metrics provided.
2. **Condition Overview** – What the diagnosed condition is, pathophysiology, and prevalence.
3. **Symptoms & Clinical Presentation** – Typical symptoms, how they manifest, severity indicators.
4. **Risk Factors** – Lifestyle, environmental, genetic, and age-related risk factors.
5. **Recommended Diagnostic Tests** – Lab work, imaging, spirometry, etc.
6. **Treatment Plan** – Medications, therapies, lifestyle modifications, and timeline.
7. **Lifestyle Recommendations** – Diet, exercise, smoking cessation, environmental adjustments.
8. **Prognosis & Follow-up** – Expected outcomes, monitoring schedule, red flags.
9. **Emergency Warning Signs** – When to seek immediate medical attention.

Format the report in clean Markdown with clear headings.
Be medically accurate but understandable to a patient.
If patient demographics are not provided, focus on the condition itself.
Always include a disclaimer that this is AI-generated and not a substitute for professional medical advice."""

    user_prompt = f"""Generate a comprehensive patient report for the following:

{patient_context}

Please provide a thorough, professional medical report."""

    try:
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_completion_tokens=4096,
        )

        report_text = response.choices[0].message.content

        return {
            "disease": req.disease.value,
            "patient_info": {
                "age": req.age,
                "height": req.height,
                "weight": req.weight,
            },
            "report": report_text,
            "model": GROQ_MODEL,
            "tokens_used": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            },
        }

    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(exc)}",
        ) from exc
