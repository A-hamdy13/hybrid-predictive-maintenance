"""LLM explanation of the structured decision output (no maintenance decisions by LLM)."""
from typing import Optional

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL
from schemas import DecisionOutput, UserLocation, VehicleIdentity


def generate_explanation(
    vehicle: VehicleIdentity,
    decision: DecisionOutput,
    user_location: Optional[UserLocation] = None,
    language: str = "en",
    *,
    api_key: Optional[str] = None,
    model: str = OPENAI_MODEL,
) -> str:
    """Turn structured decision output into a short, user-friendly explanation.
    The LLM must only explain the given decision, not suggest different actions.
    """
    client = OpenAI(api_key=api_key or OPENAI_API_KEY)

    parts = [
        f"Vehicle: {vehicle.make} {vehicle.model} {vehicle.year}.",
        f"Overall: {decision.overall_priority} – {decision.summary}",
        "Per component (schedule-based):",
    ]
    for c in decision.components:
        parts.append(f"  - {c.component}: {c.status} ({c.priority}) – {c.reason}")
    if decision.ml_output and decision.ml_output.enabled:
        m = decision.ml_output
        overall_pct = (
            m.overall_risk_score * 100 if m.overall_risk_score is not None else None
        )
        fail_pct = (
            m.failure_probability * 100 if m.failure_probability is not None else None
        )
        parts.append(
            "Telemetry (ML) risk (use this order when mentioning which components have higher risk):"
        )
        parts.append(
            f"  - Overall risk: {overall_pct:.1f}%, failure probability: {fail_pct:.1f}%."
            if overall_pct is not None and fail_pct is not None
            else "  - (no overall/failure values)"
        )
        if m.component_risks:
            sorted_risks = sorted(
                [
                    (comp, score)
                    for comp, score in m.component_risks.items()
                    if comp != "overall" and score is not None
                ],
                key=lambda x: -x[1],
            )
            for comp, score in sorted_risks:
                parts.append(
                    f"  - {comp}: {score * 100:.1f}% (higher % = higher telemetry risk)"
                )
    else:
        parts.append("Telemetry (ML): not used (recommendation from rules only).")

    language_name = {"en": "English", "ar": "Arabic", "hi": "Hindi"}.get((language or "en").lower(), "English")

    system = """You explain vehicle maintenance recommendations in plain language. You must ONLY describe and explain the already-computed recommendation. Do NOT suggest different maintenance actions or override the given status.
When both schedule-based status and telemetry (ML) risk are present, integrate both in one coherent explanation. Go into a little more detail: say what is overdue or due soon from the schedule, then explain what the telemetry model shows—overall risk and which components have elevated risk. When mentioning which components have higher telemetry risk, use the order given in the data (components listed with higher % first). Do not say brakes or engine have the highest risk unless the numbers actually show that; follow the listed order. Weave the ML risk into the narrative (e.g. "Telemetry suggests high overall risk, with elevated risk for engine oil, then brakes, then battery" only if that matches the percentages). Keep the response to 4-6 short sentences, friendly and clear."""

    user = (
        "Turn this structured output into a natural-language explanation for the driver. "
        f"Respond in {language_name}. "
        "Integrate schedule and telemetry (ML) into one flowing explanation; use the component risk order given:\n\n"
        + "\n".join(parts)
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return (resp.choices[0].message.content or decision.summary).strip()
