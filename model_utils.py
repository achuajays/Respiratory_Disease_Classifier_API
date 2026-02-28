"""
model_utils.py
--------------
Preprocessing pipeline for respiratory audio classification.
Faithfully mirrors the classes used in the training notebook
(respiratory_disease_rf_cv_91_f1_score.py) so that inference
produces exactly the same feature DataFrame the model was trained on.

DO NOT retrain the model here. Inference-only.
"""

import os

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# 1.  AudioLoader
#     Input : list[str]   — absolute file paths
#     Output: dict        — {filename: {'data': np.ndarray, 'sample_rate': int}}
# ---------------------------------------------------------------------------

class AudioLoader(BaseEstimator, TransformerMixin):
    """Load raw audio waveforms from a list of file paths."""

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        result = {}
        for file_path in X:
            # Support both Windows and POSIX paths
            filename = file_path.split("\\")[-1] if "\\" in file_path else file_path.split("/")[-1]
            y_audio, sr = librosa.load(file_path, mono=True)
            result[filename] = {"data": y_audio, "sample_rate": sr}
        return result


# ---------------------------------------------------------------------------
# 2.  AudioTrimmer
#     Trims (or zero-pads) every clip to a fixed duration so all feature
#     matrices have identical shapes.  The default duration matches the
#     shortest clip in the training dataset.
#     Input : dict — {filename: {'data', 'sample_rate'}}
#     Output: dict — same structure, data trimmed/padded
# ---------------------------------------------------------------------------

class AudioTrimmer(BaseEstimator, TransformerMixin):
    """Trim (or zero-pad) audio to a fixed duration."""

    def __init__(self, target_duration: float = 7.8560090702947845):
        self.target_duration = target_duration

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        trimmed = {}
        for filename, audio_info in X.items():
            target_samples = int(self.target_duration * audio_info["sample_rate"])

            if len(audio_info["data"]) < target_samples:
                # Pad with zeros if the clip is shorter than target
                trimmed_data = np.pad(
                    audio_info["data"],
                    (0, target_samples - len(audio_info["data"])),
                    "constant",
                )
            else:
                trimmed_data = audio_info["data"][:target_samples]

            trimmed[filename] = {
                "data": trimmed_data,
                "sample_rate": audio_info["sample_rate"],
                "duration": self.target_duration,
            }
        return trimmed


# ---------------------------------------------------------------------------
# 3.  FeatureExtractor
#     Extracts the same 8 librosa features used during training.
#     Input : dict — {filename: {'data', 'sample_rate', ...}}
#     Output: dict — {filename: {feature_name: np.ndarray, ...}}
# ---------------------------------------------------------------------------

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract the 8 acoustic features used during training."""

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        features = {}
        for filename, audio_info in X.items():
            y_audio = audio_info["data"]
            sr = audio_info["sample_rate"]

            features[filename] = {
                "chroma_stft":        librosa.feature.chroma_stft(y=y_audio, sr=sr),
                "mfcc":               librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13),
                "mel_spectrogram":    librosa.feature.melspectrogram(y=y_audio, sr=sr),
                "spectral_contrast":  librosa.feature.spectral_contrast(y=y_audio, sr=sr),
                "spectral_centroid":  librosa.feature.spectral_centroid(y=y_audio, sr=sr),
                "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y_audio, sr=sr),
                "spectral_rolloff":   librosa.feature.spectral_rolloff(y=y_audio, sr=sr),
                "zero_crossing_rate": librosa.feature.zero_crossing_rate(y=y_audio),
            }
        return features


# ---------------------------------------------------------------------------
# 4.  FeatureStatisticsCalculator
#     Collapses per-frame arrays → mean / std / max / min.
#     Drops the two features excluded during training:
#       - mel_spectrogram_min
#       - chroma_stft_max
#     Also drops the non-numeric 'filename' column before returning.
#     Input : dict — {filename: {feature_name: np.ndarray}}
#     Output: pd.DataFrame — one row per sample, numeric columns only
# ---------------------------------------------------------------------------

class FeatureStatisticsCalculator(BaseEstimator, TransformerMixin):
    """Compute mean/std/max/min statistics and return a numeric DataFrame."""

    def __init__(self, excluded_features=None):
        # Match the exact exclusions applied during training
        self.excluded_features = excluded_features or ["mel_spectrogram_min", "chroma_stft_max"]

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        feature_stats = []
        for filename, features in X.items():
            file_stats = {"filename": filename}

            for feature_name, feature_data in features.items():
                file_stats[f"{feature_name}_mean"] = np.mean(feature_data)
                file_stats[f"{feature_name}_std"]  = np.std(feature_data)
                file_stats[f"{feature_name}_max"]  = np.max(feature_data)
                file_stats[f"{feature_name}_min"]  = np.min(feature_data)

            feature_stats.append(file_stats)

        df = pd.DataFrame(feature_stats)

        # Drop the two features excluded during training
        for col in self.excluded_features:
            if col in df.columns:
                df = df.drop(col, axis=1)

        # Return only numeric columns (matches training: df.select_dtypes(exclude=['object']))
        return df.select_dtypes(exclude=["object"])


# ---------------------------------------------------------------------------
# 5.  Pipeline factory  (matches create_respiratory_pipeline() from notebook)
# ---------------------------------------------------------------------------

def create_respiratory_pipeline() -> Pipeline:
    """
    Return a fresh sklearn Pipeline:
      1. AudioLoader              — loads WAV → dict
      2. AudioTrimmer             — trims/pads to 7.856 s
      3. FeatureExtractor         — 8 librosa features per file
      4. FeatureStatisticsCalculator — mean/std/max/min → DataFrame

    Usage
    -----
    pipeline = create_respiratory_pipeline()
    features_df = pipeline.transform(["/path/to/audio.wav"])
    prediction  = model.predict(features_df)
    """
    excluded_features = ["mel_spectrogram_min", "chroma_stft_max"]

    return Pipeline(
        steps=[
            ("load_audio",          AudioLoader()),
            ("trim_audio",          AudioTrimmer(target_duration=7.8560090702947845)),
            ("extract_features",    FeatureExtractor()),
            ("calculate_statistics", FeatureStatisticsCalculator(excluded_features=excluded_features)),
        ]
    )


# ---------------------------------------------------------------------------
# 6.  Convenience predict function  (mirrors predict_respiratory_condition)
# ---------------------------------------------------------------------------

def predict_respiratory_condition(
    wav_file_path: str,
    model_path: str = "respiratory_classifier.pkl",
) -> dict:
    """
    Predict respiratory condition from a WAV file using the saved model.

    Parameters
    ----------
    wav_file_path : str  — path to the .wav file
    model_path    : str  — path to the .pkl model file

    Returns
    -------
    dict with keys: prediction, probability, all_probabilities
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    model = joblib.load(model_path)
    pipeline = create_respiratory_pipeline()

    features_df = pipeline.transform([wav_file_path])

    prediction    = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]

    return {
        "prediction":       prediction,
        "probability":      float(np.max(probabilities)),
        "all_probabilities": dict(zip(model.classes_, probabilities)),
    }
