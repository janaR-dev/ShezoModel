"""Optional LLM narration with a hard offline fallback.

Two rules make this safe to use with a real model:

1. The model only *writes prose*. It never sets the verdict, confidence or risk.
2. Any failure (no key, timeout, bad JSON, unsafe phrasing) falls back to the
   deterministic narrator, which always contains the disclosure sentence.

The client is plain stdlib ``urllib`` so the package has no dependencies.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from ..models import Modality, RiskLevel, Verdict
from .claims import Claim
from .safety import DISCLOSURE

SYSTEM_PROMPT = """You are the narration layer of a grounding tool used by people who experience psychosis, \
alongside their care team. You receive a structured record: the patient's own words, what a camera/microphone \
did or did not pick up, a sensor-evidence verdict, and a risk level.

Absolute rules:
- Never diagnose. Never say the person is hallucinating, delusional, psychotic, or "wrong".
- Never assert that what they experienced is not real. Speak only about sensor evidence.
- Use phrases like "the camera didn't pick up...", "the microphone recorded no speech", "I can't confirm".
- Do not argue with the experience, do not debate whether it is true, do not pile on reassurance.
- Be warm, short, concrete: 2-4 sentences, second person, present tense, no medical advice.
- Always leave the decision to a human clinician or care team.
- If risk level is high or critical, prioritise contacting a real person or emergency services.

Return ONLY JSON: {"headline": "<max 12 words>", "explanation": "<2-4 sentences>"}"""


class LLMClient(Protocol):
    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...


class OpenAICompatibleLLM:
    """Minimal chat-completions client (OpenAI, Groq, OpenRouter, Ollama...)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = 20.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0.3,
            "max_tokens": 400,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


def build_default_llm() -> Optional[LLMClient]:
    """Build a client from env vars, or return None to stay fully offline."""
    key = os.environ.get("SHEZO_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAICompatibleLLM(
        api_key=key,
        base_url=os.environ.get("SHEZO_LLM_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("SHEZO_LLM_MODEL", "gpt-4o-mini"),
    )


_HEADLINES = {
    Verdict.CORROBORATED: "The camera picked something up that matches what you described.",
    Verdict.NO_SENSOR_EVIDENCE: "The camera was on, and it didn't pick that up this time.",
    Verdict.INCONCLUSIVE: "I couldn't get a usable reading this time.",
}


@dataclass
class Narration:
    headline: str
    explanation: str
    source: str  # "llm" or "offline"


def _offline(claim: Claim, verdict: Verdict, risk: RiskLevel, sensor_summary: str) -> Narration:
    headline = _HEADLINES[verdict]
    lines = []
    if claim.text:
        quoted = claim.text.strip()
        if quoted.endswith((".", "!", "?")):
            quoted = quoted[:-1]
        spoke = f'You told me: "{quoted}".'
    else:
        spoke = "You asked me to check."

    if verdict is Verdict.CORROBORATED:
        lines.append(spoke)
        lines.append(f"{sensor_summary} So there is something there to look at together.")
    elif verdict is Verdict.NO_SENSOR_EVIDENCE:
        lines.append(spoke)
        lines.append(
            f"{sensor_summary} I can only tell you what that one frame showed. I can't tell you what you "
            "experienced was or wasn't real, and I'm not going to pretend otherwise."
        )
    else:
        lines.append(spoke)
        lines.append(
            f"{sensor_summary} That means I have nothing reliable to offer either way, so let's not treat it "
            "as settled."
        )

    if risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        lines.append("This feels like a lot right now. Your care team is being told, and staying with a real person matters more than my reading.")
    else:
        lines.append("Let's do one grounding step and you decide with your care team what it meant.")

    return Narration(headline=headline, explanation=" ".join(lines), source="offline")


def narrate(
    claim: Claim,
    verdict: Verdict,
    risk: RiskLevel,
    sensor_summary: str,
    llm: Optional[LLMClient] = None,
) -> Narration:
    fallback = _offline(claim, verdict, risk, sensor_summary)
    if llm is None:
        return fallback

    payload = {
        "patient_words": claim.text,
        "modality": claim.modality.value,
        "verdict": verdict.value,
        "risk_level": risk.value,
        "sensor_evidence": sensor_summary,
        "recognised_entities": claim.entities,
    }
    try:
        raw = llm.complete(
            SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False), json_mode=True
        )
        data = json.loads(raw)
        headline = str(data.get("headline", "")).strip()
        explanation = str(data.get("explanation", "")).strip()
        if not headline or not explanation or len(explanation) > 900:
            return fallback
        lowered = explanation.lower()
        banned = ("you are hallucinating", "it is not real", "you are delusional", "you are psychotic")
        if any(phrase in lowered for phrase in banned):
            return fallback
        if DISCLOSURE not in explanation:
            explanation = explanation.rstrip() + " " + DISCLOSURE
        return Narration(headline=headline, explanation=explanation, source="llm")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, TypeError):
        return fallback
