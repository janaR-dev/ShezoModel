"""Fuse a structured claim with sensor observations into a verdict.

The math is intentionally conservative and capped: confidence never exceeds 0.90,
because no camera or microphone can settle a perceptual experience. What the
numbers express is *coverage-adjusted sensor evidence*, nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..models import Modality, Verdict
from .claims import Claim
from .perception import AudioObservation, VisualObservation

MAX_CONFIDENCE = 0.90

_LABEL_ALIASES = {
    "person": ("person", "people", "human", "man", "woman", "child", "face", "hand"),
    "animal": ("animal", "dog", "cat", "rat", "bird", "spider", "insect", "bug"),
    "camera": ("camera", "webcam", "lens", "phone"),
    "phone": ("phone", "smartphone", "device"),
    "vehicle": ("vehicle", "car", "van", "truck"),
    "object": ("object", "box", "bag", "chair", "door", "window", "mirror", "tv"),
}


#: Targets specific enough to match a single detector label. A generic
#: "object" target must never be satisfied by an unrelated background label.
SPECIFIC_TARGETS = {"person", "animal", "camera", "phone", "vehicle"}


def _canonical(label: str) -> str:
    """Map a detector label onto the canonical entity the claim layer uses."""
    lowered = label.lower()
    for canonical, aliases in _LABEL_ALIASES.items():
        if lowered == canonical or lowered in aliases:
            return canonical
    return lowered


def match(claim: Claim, visual: VisualObservation) -> tuple[bool, float]:
    """Return (matched, best detector score) for the claim's targets.

    A specific target (person, animal, camera...) must be matched specifically:
    detecting a chair must never corroborate a report of a figure.
    """
    if not visual.available or not visual.detected:
        return False, 0.0
    targets = set(claim.sensor_target)
    if not targets:
        # Nothing recognisable was described: any confident detection counts.
        return bool(visual.detected), visual.max_score
    specific = targets & SPECIFIC_TARGETS
    wanted = specific or targets
    for detected in visual.detected:
        if _canonical(detected) in wanted:
            return True, visual.max_score
    return False, 0.0


@dataclass
class Assessment:
    verdict: Verdict
    confidence: float
    sensor_summary: str
    rationale: str

    def public(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 3),
            "sensor_summary": self.sensor_summary,
            "rationale": self.rationale,
        }


def _cap(value: float) -> float:
    return round(max(0.0, min(MAX_CONFIDENCE, value)), 3)


def assess(
    claim: Claim, visual: VisualObservation, audio: AudioObservation
) -> Assessment:
    parts: List[str] = [visual.summary, audio.summary]

    if claim.modality is Modality.AUDITORY:
        if not audio.available:
            return Assessment(
                Verdict.INCONCLUSIVE,
                _cap(0.28),
                " ".join(parts),
                "Microphone was not available, so the report could not be tested against a recording.",
            )
        # Loudness is not a voice. A television, music, traffic or a fan crosses any
        # energy threshold, so only recognised speech content (a transcript) or
        # diarisation ("another speaker") may corroborate a report of voices. A bare
        # ``speech_detected`` flag from an energy heuristic never can.
        if audio.other_speaker or audio.transcript.strip():
            return Assessment(
                Verdict.CORROBORATED,
                _cap(0.55 + 0.25 * audio.coverage),
                " ".join(parts),
                "The microphone recorded speech "
                + ("from another speaker " if audio.other_speaker else "")
                + "during the window, so sound a bystander could have heard was present in the room.",
            )
        if audio.speech_detected:
            return Assessment(
                Verdict.INCONCLUSIVE,
                _cap(0.40 + 0.10 * audio.coverage),
                " ".join(parts),
                "The microphone recorded sound, but the device could not identify speech in it. "
                "A TV, music or the street sounds the same, so this reading neither corroborates "
                "nor contradicts what you heard.",
            )
        return Assessment(
            Verdict.NO_SENSOR_EVIDENCE,
            _cap(0.42 + 0.18 * audio.coverage),
            " ".join(parts),
            "The microphone recorded a usable window and no speech from another person. Voices only you can "
            "hear cannot be recorded, so this does not settle what you experienced.",
        )

    if claim.modality is Modality.VISUAL or claim.modality is Modality.MIXED:
        if not visual.available:
            if claim.modality is Modality.MIXED and audio.available:
                return Assessment(
                    Verdict.INCONCLUSIVE,
                    _cap(0.32),
                    " ".join(parts),
                    "The camera could not testify this time; the microphone window was usable.",
                )
            return Assessment(
                Verdict.INCONCLUSIVE,
                _cap(0.26),
                " ".join(parts),
                "The camera gave us nothing usable (too dark, too blurry, stale or off), so there is no evidence "
                "either way.",
            )
        matched, score = match(claim, visual)
        if matched:
            return Assessment(
                Verdict.CORROBORATED,
                _cap(0.60 + 0.30 * min(1.0, score or 0.5)),
                " ".join(parts),
                "The camera detected something matching your description, so what you described is present in the frame.",
            )
        return Assessment(
            Verdict.NO_SENSOR_EVIDENCE,
            _cap(0.42 + 0.34 * visual.coverage),
            " ".join(parts),
            "The camera was on and usable, and its detector found nothing matching your description in that frame. "
            "That is one frame, not proof about your experience.",
        )

    return Assessment(
        Verdict.INCONCLUSIVE,
        _cap(0.22),
        " ".join(parts),
        "The report was not something the camera or microphone can test, so no verdict is offered.",
    )
