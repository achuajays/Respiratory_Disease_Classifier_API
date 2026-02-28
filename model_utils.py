"""
model_utils.py
--------------
Preprocessing pipeline for respiratory audio classification.
Faithfully mirrors the classes used in the training notebook
(respiratory_disease_rf_cv_91_f1_score.py) so that inference
produces exactly the same feature DataFrame the model was trained on.

DO NOT retrain the model here. Inference-only.
"""

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
            filename = file_path.split("\\")[-1] if "\\" in file_path else file_path.split("/")[-1]
            y_audio, sr = librosa.load(file_path, mono=True)
            result[filename] = {"data": y_audio, "sample_rate": sr}
        return result


# ---------------------------------------------------------------------------
# 2.  AudioTrimmer
#     Trims (or zero-pads) every clip to a fixed duration so all feature
#     matrices have identical shapes.  The default duration matches the
#     shortest clip in the training dataset.
# ---------------------------------------------------------------------------

class AudioTrimmer(BaseEstimator, TransformerMixin):
    """Trim (or zero-pad) audio to a fixed duration."""

    TARGET_DURATION = 7.8560090702947845

    def __init__(self, target_duration: float = TARGET_DURATION):
        self.target_duration = target_duration

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        trimmed = {}
        for filename, audio_info in X.items():
            target_samples = int(self.target_duration * audio_info["sample_rate"])

            if len(audio_info["data"]) < target_samples:
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
#     Drops excluded features and non-numeric columns.
# ---------------------------------------------------------------------------

_DEFAULT_EXCLUDED = ("mel_spectrogram_min", "chroma_stft_max")


class FeatureStatisticsCalculator(BaseEstimator, TransformerMixin):
    """Compute mean/std/max/min statistics and return a numeric DataFrame."""

    def __init__(self, excluded_features=None):
        self.excluded_features = excluded_features or list(_DEFAULT_EXCLUDED)

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        rows = []
        for filename, features in X.items():
            row = {"filename": filename}
            for feat_name, feat_data in features.items():
                row[f"{feat_name}_mean"] = np.mean(feat_data)
                row[f"{feat_name}_std"]  = np.std(feat_data)
                row[f"{feat_name}_max"]  = np.max(feat_data)
                row[f"{feat_name}_min"]  = np.min(feat_data)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Drop excluded columns
        df = df.drop(
            columns=[c for c in self.excluded_features if c in df.columns],
        )

        # Return only numeric columns (drops 'filename')
        return df.select_dtypes(exclude=["object"])


# ---------------------------------------------------------------------------
# 5.  Pipeline factory
# ---------------------------------------------------------------------------

def create_respiratory_pipeline() -> Pipeline:
    """
    Return a fresh sklearn Pipeline:
      1. AudioLoader              — loads WAV → dict
      2. AudioTrimmer             — trims/pads to 7.856 s
      3. FeatureExtractor         — 8 librosa features per file
      4. FeatureStatisticsCalculator — mean/std/max/min → DataFrame
    """
    return Pipeline(
        steps=[
            ("load_audio",           AudioLoader()),
            ("trim_audio",           AudioTrimmer()),
            ("extract_features",     FeatureExtractor()),
            ("calculate_statistics", FeatureStatisticsCalculator()),
        ]
    )
