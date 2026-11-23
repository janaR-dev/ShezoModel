"""Camera and microphone perception layer.

The app never sends raw media to this package. A device-side helper runs its own
detector/anomaly model and posts a *sample* of derived numbers and labels. The
analyzers below turn that into an observation the engine can reason about, and
they are honest when coverage is missing.

To plug in a real detector (YOLO, CLIP, an on-device model), implement
:class:`FrameAnalyzer` and pass it to the engine. Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from ..models import utcnow

MIN_CONFIDENT_LABEL = 0.35

#: Hard usability gates. A frame that fails any one of them cannot testify about
#: the room: at 5% mean brightness a dark corner and a black image are
#: indistinguishable, so reporting "the detector found nothing" would be a claim
#: the frame cannot support. These checks are explicit rather than folded into the
#: coverage score so a single bad property cannot be averaged away.
MIN_USABLE_BRIGHTNESS = 0.12
MAX_USABLE_BLUR = 0.75
MAX_FRAME_AGE_S = 5.0
MIN_USABLE_COVERAGE = 0.25


@dataclass
class FrameSample:
    """One camera frame as summarised by the patient's device.

    ``labels`` maps detector labels to confidence, e.g. ``{"person": 0.91}``.
    ``image_ref`` is only an opaque handle; the model never stores pixels.
    """

    labels: Dict[str, float] = field(default_factory=dict)
    faces: int = 0
    motion: float = 0.0        # 0..1 mean absolute frame difference
    brightness: float = 0.5    # 0..1
    blur: float = 0.0          # 0..1, higher means blurrier
    seconds_since_frame: float = 0.0
    image_ref: Optional[str] = None
    captured_at: str = field(default_factory=utcnow)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioSample:
    """One microphone window as summarised by the patient's device."""

    transcript: str = ""
    duration_s: float = 3.0
    ambient_db: float = 35.0
    speech_detected: bool = False
    other_speaker: bool = False   # diarisation: someone besides the patient
    snr_db: float = 12.0
    captured_at: str = field(default_factory=utcnow)


@dataclass
class VisualObservation:
    available: bool = False
    coverage: float = 0.0          # 0..1 how much of the frame was usable
    detected: List[str] = field(default_factory=list)
    max_score: float = 0.0
    faces: int = 0
    motion: float = 0.0
    summary: str = "No camera coverage."

    def public(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "coverage": round(self.coverage, 2),
            "detected": list(self.detected),
            "faces": self.faces,
            "motion": round(self.motion, 2),
            "summary": self.summary,
        }


@dataclass
class AudioObservation:
    available: bool = False
    coverage: float = 0.0
    speech_detected: bool = False
    other_speaker: bool = False
    ambient_db: float = 0.0
    transcript: str = ""
    summary: str = "No microphone coverage."

    def public(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "coverage": round(self.coverage, 2),
            "speech_detected": self.speech_detected,
            "other_speaker": self.other_speaker,
            "ambient_db": round(self.ambient_db, 1),
            "transcript": self.transcript,
            "summary": self.summary,
        }


class FrameAnalyzer(Protocol):
    def analyze(self, sample: Optional[FrameSample]) -> VisualObservation: ...


class AudioAnalyzer(Protocol):
    def analyze(self, sample: Optional[AudioSample]) -> AudioObservation: ...


def _coverage_from_frame(sample: FrameSample) -> float:
    """Usable coverage: dark, blurry or stale frames cannot testify about a room."""
    if sample.seconds_since_frame > MAX_FRAME_AGE_S:
        return 0.0
    coverage = 0.55
    if sample.brightness >= 0.18:
        coverage += 0.2
    else:
        coverage -= 0.25
    if sample.blur <= 0.5:
        coverage += 0.15
    else:
        coverage -= 0.1
    if sample.motion > 0.9:
        coverage -= 0.1
    return max(0.0, min(1.0, coverage))


def _unusable_reason(sample: FrameSample, coverage: float) -> str:
    """Why this frame cannot testify, or an empty string when it can."""
    if sample.seconds_since_frame > MAX_FRAME_AGE_S:
        return f"the last frame was {sample.seconds_since_frame:.0f}s old"
    if sample.brightness < MIN_USABLE_BRIGHTNESS:
        return f"the frame was too dark (brightness {sample.brightness:.2f})"
    if sample.blur > MAX_USABLE_BLUR:
        return f"the frame was too blurry (blur {sample.blur:.2f})"
    if coverage < MIN_USABLE_COVERAGE:
        return f"only {coverage:.2f} of the frame was usable"
    return ""


class MetadataFrameAnalyzer:
    """Trusts the device detector's labels; refuses to testify when unusable."""

    def __init__(self, min_score: float = MIN_CONFIDENT_LABEL) -> None:
        self.min_score = min_score

    def analyze(self, sample: Optional[FrameSample]) -> VisualObservation:
        if sample is None:
            return VisualObservation(available=False, coverage=0.0, summary="Camera was off for this check.")
        coverage = _coverage_from_frame(sample)
        reason = _unusable_reason(sample, coverage)
        if reason:
            return VisualObservation(
                available=False,
                coverage=coverage,
                summary=(
                    f"Camera image could not be read: {reason}. "
                    "This check has no usable camera evidence either way."
                ),
            )
        detected = sorted(
            label for label, score in (sample.labels or {}).items() if score >= self.min_score
        )
        top = max((sample.labels or {}).values(), default=0.0)
        if detected:
            summary = "Camera saw: " + ", ".join(detected) + f" (confidence up to {top:.0%})."
        else:
            summary = "Camera was on and the frame was usable, but the detector recognised no person or object of interest."
        return VisualObservation(
            available=True,
            coverage=coverage,
            detected=detected,
            max_score=float(top),
            faces=sample.faces,
            motion=sample.motion,
            summary=summary,
        )


class MicrophoneAnalyzer:
    def analyze(self, sample: Optional[AudioSample]) -> AudioObservation:
        if sample is None:
            return AudioObservation(available=False, summary="Microphone was off for this check.")
        if sample.duration_s < 0.8:
            return AudioObservation(available=False, coverage=0.2, summary="Audio clip was too short to analyse.")
        coverage = 0.85 if sample.snr_db >= 5 else 0.45
        if sample.other_speaker:
            summary = "Microphone recorded another speaker talking during the check."
        elif sample.transcript.strip():
            summary = "Microphone recorded speech during the check (a single nearby speaker)."
        elif sample.speech_detected:
            summary = (
                "Microphone recorded sound during the check, but the device could not identify "
                "speech in it — a TV, music or traffic sounds the same."
            )
        else:
            summary = (
                f"Microphone recorded no speech, only room tone around {sample.ambient_db:.0f} dB."
            )
        return AudioObservation(
            available=True,
            coverage=coverage,
            speech_detected=sample.speech_detected,
            other_speaker=sample.other_speaker,
            ambient_db=sample.ambient_db,
            transcript=sample.transcript,
            summary=summary,
        )


class NullFrameAnalyzer:
    def analyze(self, sample: Optional[FrameSample]) -> VisualObservation:
        return VisualObservation(available=False, summary="Camera not used.")


class NullAudioAnalyzer:
    def analyze(self, sample: Optional[AudioSample]) -> AudioObservation:
        return AudioObservation(available=False, summary="Microphone not used.")


def frame_from_dict(data: Optional[Dict[str, Any]], consent: bool) -> Optional[FrameSample]:
    """Build a frame sample from an API payload, enforcing session consent."""
    if not data or not consent:
        return None
    labels = {str(k): float(v) for k, v in (data.get("labels") or {}).items()}
    return FrameSample(
        labels=labels,
        faces=int(data.get("faces") or 0),
        motion=float(data.get("motion") or 0.0),
        brightness=float(data.get("brightness", 0.5)),
        blur=float(data.get("blur") or 0.0),
        seconds_since_frame=float(data.get("seconds_since_frame") or 0.0),
        image_ref=data.get("image_ref"),
        extra={k: v for k, v in data.items() if k not in {"labels", "faces", "motion", "brightness", "blur", "seconds_since_frame", "image_ref"}},
    )


def audio_from_dict(data: Optional[Dict[str, Any]], consent: bool) -> Optional[AudioSample]:
    if not data or not consent:
        return None
    return AudioSample(
        transcript=str(data.get("transcript") or ""),
        duration_s=float(data.get("duration_s") or 3.0),
        ambient_db=float(data.get("ambient_db", 35.0)),
        speech_detected=bool(data.get("speech_detected")),
        other_speaker=bool(data.get("other_speaker")),
        snr_db=float(data.get("snr_db", 12.0)),
    )
