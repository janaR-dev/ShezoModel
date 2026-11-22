"""AI layer for Shezo: perception, claim parsing, scoring, safety, narration."""

from .claims import Claim, parse_claim
from .engine import CheckOutcome, GroundingEngine
from .grounding import STEP_LIBRARY, select_steps
from .llm import LLMClient, Narration, OpenAICompatibleLLM, build_default_llm, narrate
from .perception import (
    AudioObservation,
    AudioSample,
    FrameSample,
    MetadataFrameAnalyzer,
    MicrophoneAnalyzer,
    VisualObservation,
    audio_from_dict,
    frame_from_dict,
)
from .safety import CRISIS_RESOURCES, DISCLOSURE, SafetyAssessment, screen
from .scoring import Assessment, assess

__all__ = [
    "Claim",
    "parse_claim",
    "CheckOutcome",
    "GroundingEngine",
    "STEP_LIBRARY",
    "select_steps",
    "LLMClient",
    "Narration",
    "OpenAICompatibleLLM",
    "build_default_llm",
    "narrate",
    "AudioObservation",
    "AudioSample",
    "FrameSample",
    "MetadataFrameAnalyzer",
    "MicrophoneAnalyzer",
    "VisualObservation",
    "audio_from_dict",
    "frame_from_dict",
    "CRISIS_RESOURCES",
    "DISCLOSURE",
    "SafetyAssessment",
    "screen",
    "Assessment",
    "assess",
]
