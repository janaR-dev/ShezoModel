"""Deterministic risk screening.

This module is the safety floor. It runs before any model call and its result can
only be *raised* by later stages, never lowered. An LLM is never allowed to
decide that a message is fine.

Nothing here diagnoses. It flags language that warrants a human response and
names the human response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..models import RiskLevel

CRISIS_RESOURCES = (
    "If you might act on these thoughts, contact your local emergency number now, "
    "or a crisis line (US 988, UK 111 option 2, EU 112)."
)

DISCLOSURE = (
    "Shezo reports what the camera and microphone did or did not pick up. It does not diagnose, "
    "and it cannot confirm or rule out a hallucination. A clinician or your care team decides."
)


@dataclass
class SafetyAssessment:
    risk_level: RiskLevel
    flags: List[str] = field(default_factory=list)
    crisis_protocol: bool = False
    notify_emergency_contact: bool = False
    note: str = ""

    def public(self) -> dict:
        return {
            "risk_level": self.risk_level.value,
            "flags": list(self.flags),
            "crisis_protocol": self.crisis_protocol,
            "notify_emergency_contact": self.notify_emergency_contact,
            "note": self.note,
        }


_SELF_HARM = (
    r"\bkill myself\b", r"\bend my life\b", r"\bsuicid", r"\bwant to die\b",
    r"\bdon'?t want to live\b", r"\bhurt myself\b", r"\bcut myself\b",
    r"\boverdose\b", r"\btake (?:all )?my pills\b", r"\bjump off\b",
)

_HARM_OTHERS = (
    r"\bhurt (?:him|her|them|someone|people)\b", r"\bkill (?:him|her|them|someone)\b",
    r"\bstab\b", r"\battack (?:him|her|them)\b", r"\bthey deserve to die\b",
    r"\bhave to stop (?:him|her|them)\b",
)

_COMMAND_HALLUCINATION = (
    r"\b(?:voice|voices|god|the tv|the radio|they)\b[^.]{0,40}\b(?:tell|told|tells|telling|order|command|want)\w*\b[^.]{0,30}\b(?:me|us)\b",
    r"\btold me to\b", r"\btells me to\b", r"\bis making me\b",
    r"\bsaid (?:i|we) (?:must|have to|need to)\b",
)

_SEVERE_DISORGANISATION = (
    r"\bi (?:am|'m) (?:god|jesus|the devil)\b", r"\bthey (?:are|'re) (?:poisoning|drugging)\b",
    r"\b implanted\b", r"\bmind control\b", r"\bchips? in my (?:head|brain)\b",
    r"\b(?:3|three|several) days? (?:without|no) sleep\b", r"\bhaven'?t slept\b",
)

_MEDICATION_STOP = (
    r"\b(?:stopped|stop|quit) (?:taking )?(?:my )?(?:meds|medication|medicines|pills)\b",
    r"\bdon'?t need (?:my )?(?:meds|medication|pills)\b",
)


def _matches(patterns: tuple, text: str) -> Optional[str]:
    for pattern in patterns:
        if re.search(pattern, text):
            return pattern
    return None


def screen(text: str, distress: Optional[int] = None) -> SafetyAssessment:
    """Screen one utterance plus an optional 0-10 distress rating."""
    lowered = (text or "").lower()
    flags: List[str] = []
    level = RiskLevel.LOW

    if _matches(_SELF_HARM, lowered):
        flags.append("self_harm_language")
        level = RiskLevel.CRITICAL
    if _matches(_HARM_OTHERS, lowered):
        flags.append("harm_to_others_language")
        level = RiskLevel.CRITICAL
    if _matches(_COMMAND_HALLUCINATION, lowered):
        flags.append("command_hallucination")
        level = RiskLevel.highest(level, RiskLevel.HIGH)
    if _matches(_SEVERE_DISORGANISATION, lowered):
        flags.append("severe_disorganisation")
        level = RiskLevel.highest(level, RiskLevel.HIGH)
    if _matches(_MEDICATION_STOP, lowered):
        flags.append("medication_stopped")
        level = RiskLevel.highest(level, RiskLevel.ELEVATED)

    if distress is not None:
        if distress >= 9:
            flags.append("extreme_distress")
            level = RiskLevel.highest(level, RiskLevel.CRITICAL)
        elif distress >= 7:
            flags.append("high_distress")
            level = RiskLevel.highest(level, RiskLevel.HIGH)
        elif distress >= 5:
            flags.append("moderate_distress")
            level = RiskLevel.highest(level, RiskLevel.ELEVATED)

    crisis = level is RiskLevel.CRITICAL
    note = ""
    if crisis:
        note = CRISIS_RESOURCES
    elif level is RiskLevel.HIGH:
        note = "Your care team is being notified now; stay on the grounding screen with me."
    elif level is RiskLevel.ELEVATED:
        note = "Let's slow this down together, and I'll flag this for your care team."

    return SafetyAssessment(
        risk_level=level,
        flags=flags,
        crisis_protocol=crisis,
        notify_emergency_contact=crisis,
        note=note,
    )
