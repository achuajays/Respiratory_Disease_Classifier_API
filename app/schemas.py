"""
app.schemas
-----------
Pydantic models and enums shared across routers.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RespiratoryDisease(str, Enum):
    """The 8 respiratory conditions the model can classify."""

    asthma = "Asthma"
    bronchiectasis = "Bronchiectasis"
    bronchiolitis = "Bronchiolitis"
    copd = "COPD"
    healthy = "Healthy"
    lrti = "LRTI"
    pneumonia = "Pneumonia"
    urti = "URTI"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ReportRequest(BaseModel):
    """Request body for the /report endpoint."""

    disease: RespiratoryDisease = Field(
        ..., description="Diagnosed respiratory condition"
    )
    age: Optional[int] = Field(
        None, ge=0, le=150, description="Patient age in years"
    )
    height: Optional[float] = Field(
        None, gt=0, description="Patient height in cm"
    )
    weight: Optional[float] = Field(
        None, gt=0, description="Patient weight in kg"
    )


# ---------------------------------------------------------------------------
# Response models  (used in OpenAPI docs)
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""

    prediction: str
    confidence: float
    all_probabilities: dict[str, float]


class ReportResponse(BaseModel):
    """Response from the /report endpoint."""

    disease: str
    patient_info: dict
    report: str
    model: str
    tokens_used: dict[str, int]
