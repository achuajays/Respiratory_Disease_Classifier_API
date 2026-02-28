# 🫁 Respiratory Disease Classifier API

A FastAPI REST API that classifies respiratory conditions from breath/cough WAV audio files using a pre-trained **Random Forest** model with a custom sklearn preprocessing pipeline.

---

## Project Structure

```
Respiratory_Disease_Classifier_API/
├── main.py                      # FastAPI application
├── model_utils.py               # Preprocessing pipeline (AudioLoader → Trimmer → FeatureExtractor → Stats)
├── respiratory_classifier.pkl   # Pre-trained Random Forest model
├── requirements.txt             # Pip dependencies
└── pyproject.toml               # uv / PEP 517 metadata
```

---

## Quickstart

### 1. Install dependencies

**With pip:**
```bash
pip install -r requirements.txt
```

**With uv (recommended — uses pyproject.toml):**
```bash
uv sync
```

### 2. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open interactive docs

```
http://localhost:8000/docs
```

Upload any `.wav` breath/cough recording and inspect the prediction result directly in the Swagger UI.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/classes` | List all condition labels the model can predict |
| `POST` | `/predict` | Upload a WAV file → get prediction + confidence |

### `/predict` response example

```json
{
  "prediction": "COPD",
  "confidence": 0.87,
  "all_probabilities": {
    "Healthy": 0.05,
    "COPD": 0.87,
    "Pneumonia": 0.08
  }
}
```

---

## Preprocessing Pipeline

`model_utils.py` implements a sklearn-compatible pipeline:

1. **`AudioLoader`** — loads WAV files via `librosa.load()`
2. **`AudioTrimmer`** — removes leading/trailing silence (`librosa.effects.trim`)
3. **`FeatureExtractor`** — computes MFCCs, Chroma, Spectral Contrast, Mel Spectrogram, ZCR, RMS
4. **`FeatureStatisticsCalculator`** — collapses frame-level features into mean/std statistics → flat `pd.DataFrame`

---

## Why CPU-Only is Fine Here

- Random Forest is CPU-native — no GPU dependency
- No PyTorch / CUDA / tensor ops
- Librosa feature extraction is lightweight
- Startup time < 2 s on any modern CPU
