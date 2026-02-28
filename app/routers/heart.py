"""
app.routers.heart
-----------------
Heart disease risk assessment using a multi-step LLM chain:
  Step 1 — Triage:    urgency classification (JSON)
  Step 2 — Diagnosis: condition analysis     (JSON)
  Step 3 — Report:    full patient report    (Markdown)

All three steps run sequentially; each step receives the previous output.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.schemas import HeartAnalysisResponse, HeartDiseaseInput

router = APIRouter(prefix="/heart", tags=["Heart Disease"])

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM = """\
You are an emergency triage AI cardiologist.
Analyze the patient's clinical data and classify urgency.

Return ONLY valid JSON with this exact structure:
{
  "urgency": "critical" | "high" | "moderate" | "low",
  "reasoning": "brief explanation",
  "immediate_action_needed": true | false,
  "key_red_flags": ["flag1", "flag2"]
}

Guidelines:
- CRITICAL: ST depression > 2, ASY chest pain + low HR + Flat/Down ST slope
- HIGH: Multiple risk factors, exercise angina, abnormal ECG
- MODERATE: Some risk factors present, borderline values
- LOW: Normal ranges, no significant flags"""

DIAGNOSIS_SYSTEM = """\
You are a senior cardiologist AI.
Based on the patient data AND the triage assessment, provide a detailed diagnosis analysis.

Return ONLY valid JSON with this exact structure:
{
  "primary_condition": "most likely heart condition",
  "differential_diagnoses": [
    {"condition": "name", "likelihood": "high|medium|low", "reasoning": "why"}
  ],
  "risk_score": 1-10,
  "risk_factors_present": ["factor1", "factor2"],
  "protective_factors": ["factor1"],
  "abnormal_values": [
    {"parameter": "name", "value": "actual", "normal_range": "expected", "significance": "explanation"}
  ]
}

Consider these conditions: Coronary Artery Disease, Heart Failure, Arrhythmia,
Valvular Heart Disease, Hypertensive Heart Disease, Angina Pectoris, Myocardial Infarction risk."""

REPORT_SYSTEM = """\
You are a senior cardiologist AI assistant.
Generate a comprehensive, patient-friendly heart disease risk report in Markdown.

Use the patient data, triage result, and diagnosis analysis to create the report.

The report MUST include ALL of these sections:

1. **Patient Summary** — Demographics, vitals, and key metrics
2. **Triage Assessment** — Urgency level and immediate actions
3. **Clinical Findings** — Analysis of all provided metrics with normal ranges
4. **Diagnosis & Risk Assessment** — Primary condition, differentials, risk score
5. **Risk Factors Analysis** — Present risk factors and protective factors
6. **Recommended Tests** — ECG, Echo, Stress test, blood work, imaging
7. **Treatment Plan** — Medications, interventions, lifestyle changes
8. **Lifestyle Modifications** — Diet (DASH/Mediterranean), exercise, stress management
9. **Follow-up Schedule** — Timeline for monitoring and check-ups
10. **Emergency Warning Signs** — When to call 911

Always include a disclaimer that this is AI-generated and not a substitute for professional medical advice."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_patient_summary(req: HeartDiseaseInput) -> str:
    """Format the clinical input as a readable summary for the LLM."""
    return (
        f"Age: {req.age} | Sex: {req.sex.value}\n"
        f"Chest Pain Type: {req.chest_pain_type.value}\n"
        f"Resting BP: {req.resting_bp} mm Hg\n"
        f"Cholesterol: {req.cholesterol} mg/dl\n"
        f"Fasting Blood Sugar > 120: {'Yes' if req.fasting_bs else 'No'}\n"
        f"Resting ECG: {req.resting_ecg.value}\n"
        f"Max Heart Rate: {req.max_hr}\n"
        f"Exercise Angina: {req.exercise_angina.value}\n"
        f"Oldpeak (ST depression): {req.oldpeak}\n"
        f"ST Slope: {req.st_slope.value}"
    )


async def _llm_call(groq_client, model: str, system: str, user: str) -> str:
    """Single Groq chat completion call."""
    response = await groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
        max_completion_tokens=4096,
    )
    return response.choices[0].message.content, response.usage


def _parse_json_safe(text: str) -> dict:
    """Parse JSON from LLM output, handling markdown code fences."""
    cleaned = text.strip()
    # Strip ```json ... ``` wrapper if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    summary="Multi-step heart disease risk analysis (Triage → Diagnosis → Report)",
    response_model=HeartAnalysisResponse,
)
async def analyze_heart(request: Request, req: HeartDiseaseInput):
    """
    Submit clinical data and receive a comprehensive 3-step analysis:

    1. **Triage** — urgency classification + red flags
    2. **Diagnosis** — conditions, risk score, abnormal values
    3. **Report** — full Markdown patient report

    All powered by Groq's Llama 4 Scout model.
    """
    settings = get_settings()
    groq_client = request.app.state.groq
    model = settings.groq_model

    patient_summary = _build_patient_summary(req)
    total_tokens = {"prompt": 0, "completion": 0, "total": 0}

    try:
        # =================================================================
        # STEP 1 — Triage
        # =================================================================
        logger.info("🫀 Step 1/3: Triage")
        triage_text, usage1 = await _llm_call(
            groq_client, model, TRIAGE_SYSTEM,
            f"Patient clinical data:\n\n{patient_summary}",
        )
        total_tokens["prompt"] += usage1.prompt_tokens
        total_tokens["completion"] += usage1.completion_tokens

        try:
            triage = _parse_json_safe(triage_text)
        except json.JSONDecodeError:
            triage = {"raw_response": triage_text, "urgency": "unknown"}

        # =================================================================
        # STEP 2 — Diagnosis  (receives patient data + triage result)
        # =================================================================
        logger.info("🫀 Step 2/3: Diagnosis")
        diagnosis_text, usage2 = await _llm_call(
            groq_client, model, DIAGNOSIS_SYSTEM,
            f"Patient clinical data:\n\n{patient_summary}\n\n"
            f"Triage assessment:\n{json.dumps(triage, indent=2)}",
        )
        total_tokens["prompt"] += usage2.prompt_tokens
        total_tokens["completion"] += usage2.completion_tokens

        try:
            diagnosis = _parse_json_safe(diagnosis_text)
        except json.JSONDecodeError:
            diagnosis = {"raw_response": diagnosis_text}

        # =================================================================
        # STEP 3 — Report  (receives everything)
        # =================================================================
        logger.info("🫀 Step 3/3: Report generation")
        report_text, usage3 = await _llm_call(
            groq_client, model, REPORT_SYSTEM,
            f"Patient clinical data:\n\n{patient_summary}\n\n"
            f"Triage assessment:\n{json.dumps(triage, indent=2)}\n\n"
            f"Diagnosis analysis:\n{json.dumps(diagnosis, indent=2)}\n\n"
            f"Generate the full patient report now.",
        )
        total_tokens["prompt"] += usage3.prompt_tokens
        total_tokens["completion"] += usage3.completion_tokens

        total_tokens["total"] = total_tokens["prompt"] + total_tokens["completion"]

        logger.info("✅ Heart analysis complete — %d total tokens", total_tokens["total"])

        return {
            "patient_input": req.model_dump(),
            "triage": triage,
            "diagnosis": diagnosis,
            "report": report_text,
            "model": model,
            "tokens_used": total_tokens,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Heart analysis failed: {str(exc)}",
        ) from exc
