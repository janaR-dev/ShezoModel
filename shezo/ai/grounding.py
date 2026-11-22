"""Grounding exercises offered after a reality check.

Steps are short, concrete and sensory. They are coping tools, not treatment, and
the high-risk paths always end by putting a human in the loop.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..models import Modality, RiskLevel, Verdict

STEP_LIBRARY: Dict[str, Dict[str, Any]] = {
    "orient_54321": {
        "id": "orient_54321",
        "title": "5-4-3-2-1 anchoring",
        "seconds": 90,
        "instructions": [
            "Name 5 things you can see right now, out loud, one at a time.",
            "Name 4 things you can physically feel (chair, floor, fabric).",
            "Name 3 things you can hear in the room around you.",
            "Name 2 things you can smell, or two smells you like.",
            "Name 1 thing you can taste, or take one slow sip of water.",
        ],
        "why": "Moves attention from an internal experience to verifiable external input.",
    },
    "slow_breath": {
        "id": "slow_breath",
        "title": "Longer-exhale breathing",
        "seconds": 60,
        "instructions": [
            "Breathe in through your nose for 4 counts.",
            "Breathe out slowly through your mouth for 6-8 counts.",
            "Repeat 6 times. It is fine if your mind wanders; just come back to the count.",
        ],
        "why": "A longer exhale lowers physical arousal, which can turn the volume down on fear.",
    },
    "temperature": {
        "id": "temperature",
        "title": "Cool water reset",
        "seconds": 30,
        "instructions": [
            "Run cool (not icy) water over your wrists and hands for 30 seconds.",
            "Notice the temperature exactly, and say the word for it out loud.",
        ],
        "why": "Strong, safe physical sensation competes with distressing perception.",
    },
    "sound_anchor": {
        "id": "sound_anchor",
        "title": "Check the room's sound",
        "seconds": 45,
        "instructions": [
            "Stay still and listen without judging. List every sound you can actually locate.",
            "Say which sounds you can point to a source for, and which you cannot.",
            "If a sound has no locatable source, note it and keep it as 'unconfirmed' rather than deciding.",
        ],
        "why": "Separating located from unlocated sound avoids arguing with the experience.",
    },
    "visual_anchor": {
        "id": "visual_anchor",
        "title": "Look again, without deciding",
        "seconds": 45,
        "instructions": [
            "Look at the place you noticed something. Describe only what your eyes receive: colour, edges, movement.",
            "Move your head slightly and look again. Note anything that changes or stays the same.",
            "You do not have to decide right now. 'Unconfirmed' is a valid answer.",
        ],
        "why": "Observation without a forced verdict reduces the pressure that fuels distress.",
    },
    "contact_human": {
        "id": "contact_human",
        "title": "Put a person in the loop",
        "seconds": 60,
        "instructions": [
            "Send your caretaker or therapist one message: 'I'm having a rough moment, can you check in?'",
            "If you feel you might act on a thought, call your local emergency number now.",
            "Stay on this screen until someone replies if you can.",
        ],
        "why": "Human contact is the most reliable grounding step when risk is high.",
    },
    "delay_action": {
        "id": "delay_action",
        "title": "Do not act for 10 minutes",
        "seconds": 600,
        "instructions": [
            "Whatever the voice or the feeling is asking you to do, agree to wait 10 minutes first.",
            "Set a timer. In that time, do one grounding step and message your care team.",
            "You are not agreeing it is false. You are only delaying the action.",
        ],
        "why": "Command experiences are most dangerous when acted on immediately.",
    },
}


def select_steps(
    modality: Modality,
    verdict: Verdict,
    risk: RiskLevel,
    distress: int = 0,
) -> List[Dict[str, Any]]:
    """Pick 2-4 ordered steps appropriate to modality, verdict, risk and distress."""
    steps: List[Dict[str, Any]] = []

    if risk is RiskLevel.CRITICAL:
        steps.append(STEP_LIBRARY["contact_human"])
        steps.append(STEP_LIBRARY["delay_action"])
        steps.append(STEP_LIBRARY["slow_breath"])
    elif risk is RiskLevel.HIGH:
        steps.append(STEP_LIBRARY["contact_human"])
        steps.append(STEP_LIBRARY["slow_breath"])
        steps.append(STEP_LIBRARY["orient_54321"])
    else:
        if distress >= 6:
            steps.append(STEP_LIBRARY["slow_breath"])
        steps.append(STEP_LIBRARY["orient_54321"])
        if modality in (Modality.AUDITORY, Modality.MIXED):
            steps.append(STEP_LIBRARY["sound_anchor"])
        else:
            steps.append(STEP_LIBRARY["visual_anchor"])
        if distress >= 8:
            steps.append(STEP_LIBRARY["temperature"])

    seen = set()
    ordered: List[Dict[str, Any]] = []
    for step in steps:
        if step["id"] in seen:
            continue
        seen.add(step["id"])
        ordered.append(step)
    return ordered[:4]
