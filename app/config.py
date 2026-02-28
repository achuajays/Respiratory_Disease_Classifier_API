"""
app.config
----------
Centralized settings loaded from environment / .env file.
Uses Pydantic BaseSettings for type-safe configuration.
"""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings — automatically reads from .env file."""

    # --- Model ---
    model_path: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "respiratory_classifier.pkl"
    )

    # --- Cache ---
    cache_max_size: int = 128

    # --- Groq ---
    groq_api_key: str = ""
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
