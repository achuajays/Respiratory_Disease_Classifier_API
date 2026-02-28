"""
app.routers.drugs
-----------------
Drug interaction checker.
Analyzes a list of medications for interactions, contraindications,
and condition-specific warnings using Groq LLM.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.schemas import DrugCheckRequest, DrugCheckResponse

router = APIRouter(prefix="/drugs", tags=["Drug Interactions"])

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a clinical pharmacologist AI assistant. Analyze the provided medications for drug-drug interactions, contraindications, and safety warnings.

## Analysis Steps

1. **Identify each medication** — drug class, mechanism of action, common uses.
2. **Check pairwise interactions** — every combination of the provided drugs.
3. **Assess condition-specific risks** — if a medical condition is provided, check contraindications.
4. **Consider patient factors** — age, allergies if provided.
5. **Generate safety summary** — overall assessment.

## Response Format

You MUST respond with ONLY valid JSON in this exact structure:
{
  "interactions": [
    {
      "drug_pair": ["Drug A", "Drug B"],
      "severity": "major|moderate|minor|none",
      "type": "pharmacokinetic|pharmacodynamic|additive|synergistic",
      "description": "What happens when these drugs interact",
      "clinical_significance": "What this means for the patient",
      "management": "How to handle this interaction"
    }
  ],
  "warnings": [
    {
      "medication": "Drug name",
      "type": "contraindication|precaution|black_box|allergy",
      "description": "Warning details",
      "severity": "critical|high|moderate|low"
    }
  ],
  "safe_summary": "Overall safety assessment in 2-3 sentences",
  "medication_details": [
    {
      "name": "Drug name",
      "class": "Drug class",
      "common_uses": "What it's used for",
      "key_side_effects": ["side effect 1", "side effect 2"]
    }
  ],
  "recommendations": [
    "Recommendation 1",
    "Recommendation 2"
  ]
}

## Severity Guidelines
- **major**: Life-threatening or permanent damage risk. Avoid combination.
- **moderate**: May worsen condition or require monitoring. Use with caution.
- **minor**: Minimal clinical significance. Be aware.
- **none**: No known interaction.

Always note that this is AI-generated and pharmacist/physician review is required."""


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/check",
    summary="Check drug interactions and safety",
    response_model=DrugCheckResponse,
)
async def check_drugs(request: Request, req: DrugCheckRequest):
    """
    Submit a list of medications and receive:

    - **interactions** — pairwise drug-drug interactions with severity
    - **warnings** — contraindications, black box alerts, allergy flags
    - **safe_summary** — overall safety assessment
    - **report** — full Markdown pharmacology report

    Optionally provide `condition`, `age`, and `allergies` for targeted analysis.
    """
    settings = get_settings()
    groq_client = request.app.state.groq

    # Build user prompt
    med_list = ", ".join(req.medications)
    context_parts = [f"Medications to analyze: {med_list}"]

    if req.condition:
        context_parts.append(f"Patient's medical condition: {req.condition}")
    if req.age is not None:
        context_parts.append(f"Patient age: {req.age} years")
    if req.allergies:
        context_parts.append(f"Known drug allergies: {', '.join(req.allergies)}")

    user_prompt = "\n".join(context_parts)

    try:
        # --- Step 1: Structured JSON analysis ---
        json_response = await groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_completion_tokens=3072,
        )

        raw_json_text = json_response.choices[0].message.content

        # Parse JSON
        try:
            cleaned = raw_json_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines)
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = {
                "interactions": [],
                "warnings": [],
                "safe_summary": raw_json_text,
                "medication_details": [],
                "recommendations": [],
            }

        # --- Step 2: Human-readable report ---
        report_response = await groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical pharmacologist. Based on the drug interaction "
                        "analysis, generate a comprehensive patient-friendly Markdown report.\n\n"
                        "Include sections:\n"
                        "1. **Medication Overview** — what each drug does\n"
                        "2. **Interaction Report** — detailed interaction analysis\n"
                        "3. **Safety Warnings** — contraindications, precautions\n"
                        "4. **Recommendations** — dosage timing, monitoring needs\n"
                        "5. **Questions for Your Doctor** — suggested discussions\n\n"
                        "Always include a disclaimer about professional pharmacist review."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Patient info:\n{user_prompt}\n\n"
                        f"Analysis results:\n{json.dumps(parsed, indent=2)}\n\n"
                        "Generate the full patient report."
                    ),
                },
            ],
            temperature=0.4,
            max_completion_tokens=3072,
        )

        report_text = report_response.choices[0].message.content

        # Aggregate token usage
        total_tokens = {
            "prompt": (
                json_response.usage.prompt_tokens
                + report_response.usage.prompt_tokens
            ),
            "completion": (
                json_response.usage.completion_tokens
                + report_response.usage.completion_tokens
            ),
            "total": (
                json_response.usage.total_tokens
                + report_response.usage.total_tokens
            ),
        }

        return {
            "medications_checked": req.medications,
            "interactions": parsed.get("interactions", []),
            "warnings": parsed.get("warnings", []),
            "safe_summary": parsed.get("safe_summary", ""),
            "report": report_text,
            "model": settings.groq_model,
            "tokens_used": total_tokens,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Drug interaction check failed: {str(exc)}",
        ) from exc
