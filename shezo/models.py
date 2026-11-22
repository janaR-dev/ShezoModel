"""Core domain types for Shezo, an AI-assisted grounding platform.

The types here are deliberately plain: they are the shared vocabulary between
the three interfaces (patient, caretaker, therapist) and the AI layer.

Safety posture (important, see README):
  * A ``Verdict`` is a statement about *sensor coverage*, never a diagnosis.
  * The engine never says "that is a hallucination". It says what the camera or
    microphone did or did not pick up, then hands the decision to a human.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utcnow() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_ts(value: str) -> datetime:
    """Parse an ISO-8601 string produced by :func:`utcnow` (or a client)."""
    if not value:
        raise ValueError("empty timestamp")
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class Role(str, enum.Enum):
    PATIENT = "patient"
    CARETAKER = "caretaker"
    THERAPIST = "therapist"


class Modality(str, enum.Enum):
    """What kind of experience the patient is reporting."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    MIXED = "mixed"
    OTHER = "other"


class Verdict(str, enum.Enum):
    """Outcome of a reality check, expressed in terms of sensor evidence only."""

    #: The camera/microphone picked up something that matches the report.
    CORROBORATED = "corroborated"
    #: Sensors cannot confirm or deny (no coverage, or the sense is not recordable).
    INCONCLUSIVE = "inconclusive"
    #: Sensors had coverage for that area and detected nothing matching.
    NO_SENSOR_EVIDENCE = "no_sensor_evidence"


class RiskLevel(str, enum.Enum):
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {
            RiskLevel.LOW: 0,
            RiskLevel.ELEVATED: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3,
        }[self]

    @classmethod
    def highest(cls, *levels: "RiskLevel") -> "RiskLevel":
        return max(levels, key=lambda level: level.rank) if levels else cls.LOW


class DoseStatus(str, enum.Enum):
    PENDING = "pending"
    TAKEN = "taken"
    SKIPPED = "skipped"
    MISSED = "missed"


class ReportKind(str, enum.Enum):
    REALITY_CHECK = "reality_check"
    CRISIS_ESCALATION = "crisis_escalation"
    SESSION_SUMMARY = "session_summary"
    MEDICATION_MISSED = "medication_missed"
    THERAPIST_NOTE = "therapist_note"


@dataclass
class User:
    id: str
    email: str
    display_name: str
    role: Role
    created_at: str
    password_hash: str = ""
    salt: str = ""

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "role": self.role.value,
        }


@dataclass
class Patient:
    id: str
    user_id: str
    display_name: str
    created_at: str
    therapist_id: Optional[str] = None
    primary_caretaker_id: Optional[str] = None
    date_of_birth: Optional[str] = None
    care_notes: str = ""
    emergency_contact: str = ""

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "therapist_id": self.therapist_id,
            "primary_caretaker_id": self.primary_caretaker_id,
            "care_notes": self.care_notes,
            "emergency_contact": self.emergency_contact,
        }


@dataclass
class CareTeamMember:
    patient_id: str
    user_id: str
    role: Role
    relationship: str = ""
    display_name: str = ""


@dataclass
class Medication:
    id: str
    patient_id: str
    name: str
    dose: str
    times: List[str]  # local "HH:MM" wall-clock times
    instructions: str = ""
    active: bool = True
    created_at: str = ""

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "dose": self.dose,
            "times": list(self.times),
            "instructions": self.instructions,
            "active": self.active,
        }


@dataclass
class Dose:
    id: str
    medication_id: str
    patient_id: str
    due_at: str
    status: DoseStatus
    acted_at: Optional[str] = None
    medication_name: str = ""
    medication_dose: str = ""

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "medication": self.medication_name,
            "dose": self.medication_dose,
            "due_at": self.due_at,
            "status": self.status.value,
            "acted_at": self.acted_at,
        }


@dataclass
class Session:
    """A grounding session. Camera/mic are only used while this is open."""

    id: str
    patient_id: str
    started_at: str
    ended_at: Optional[str] = None
    distress_start: Optional[int] = None
    distress_end: Optional[int] = None
    camera_consent: bool = False
    mic_consent: bool = False
    status: str = "open"

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "distress_start": self.distress_start,
            "distress_end": self.distress_end,
            "camera_consent": self.camera_consent,
            "mic_consent": self.mic_consent,
            "status": self.status,
        }


@dataclass
class RealityCheck:
    id: str
    patient_id: str
    session_id: Optional[str]
    created_at: str
    claim: str
    modality: Modality
    verdict: Verdict
    confidence: float
    risk_level: RiskLevel
    headline: str
    explanation: str
    sensor_summary: str
    grounding_steps: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    escalated: bool = False
    disclosure: str = ""

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "session_id": self.session_id,
            "claim": self.claim,
            "modality": self.modality.value,
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 3),
            "risk_level": self.risk_level.value,
            "headline": self.headline,
            "explanation": self.explanation,
            "sensor_summary": self.sensor_summary,
            "grounding_steps": self.grounding_steps,
            "escalated": self.escalated,
            "disclosure": self.disclosure,
        }


@dataclass
class Report:
    id: str
    patient_id: str
    kind: ReportKind
    title: str
    summary: str
    author_role: Role
    author_id: str
    created_at: str
    payload: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    reality_check_id: Optional[str] = None
    urgent: bool = False
    recipients: List[Dict[str, Any]] = field(default_factory=list)

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "kind": self.kind.value,
            "title": self.title,
            "summary": self.summary,
            "author_role": self.author_role.value,
            "created_at": self.created_at,
            "payload": self.payload,
            "session_id": self.session_id,
            "reality_check_id": self.reality_check_id,
            "urgent": self.urgent,
            "recipients": self.recipients,
        }


@dataclass
class Note:
    id: str
    patient_id: str
    author_id: str
    author_role: Role
    author_name: str
    body: str
    created_at: str
    pinned: bool = False

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "author_id": self.author_id,
            "author_role": self.author_role.value,
            "author_name": self.author_name,
            "body": self.body,
            "created_at": self.created_at,
            "pinned": self.pinned,
        }
