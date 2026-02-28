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

import os
import tempfile

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from model_utils import create_respiratory_pipeline

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
# Globals  (populated on startup)
# ---------------------------------------------------------------------------

model = None
preprocessing_pipeline = None

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
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav", prefix="resp_"
        ) as tmp:
            content = await file.read()
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

        return JSONResponse(
            content={
                "prediction": str(prediction),
                "confidence": round(confidence, 6),
                "all_probabilities": all_probs,
            }
        )

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
