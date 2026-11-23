"""The grounding engine: the pipeline behind every reality check.

    utterance ─▶ deterministic risk screen ─▶ claim parse
              ─▶ camera/mic observations ─▶ sensor-evidence verdict
              ─▶ narration (LLM or offline) ─▶ grounding steps
              ─▶ persisted check + reports routed to the care team

Order matters: the safety screen runs first and its risk level can only be raised
later, never lowered. A model can never talk the system out of an escalation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..models import (
    Modality,
    Patient,
    RealityCheck,
    Report,
    RiskLevel,
    Session,
    Verdict,
    parse_ts,
    utcnow,
)
from ..storage import Repository, new_id
from . import safety
from .claims import Claim, parse_claim
from .grounding import select_steps
from .llm import LLMClient, Narration, build_default_llm, narrate
from .perception import (
    AudioAnalyzer,
    AudioSample,
    FrameAnalyzer,
    FrameSample,
    MetadataFrameAnalyzer,
    MicrophoneAnalyzer,
    VisualObservation,
    AudioObservation,
)
from .scoring import Assessment, assess


#: A session with no checks for this long is closed automatically, so a tab the
#: patient closed (or reloaded away) cannot leave an orphaned open session that
#: keeps blocking new sessions and never gets a summary.
SESSION_TIMEOUT_ENV = "SHEZO_SESSION_TIMEOUT_MINUTES"
DEFAULT_SESSION_TIMEOUT_MINUTES = 90


def session_timeout_minutes() -> int:
    raw = os.environ.get(SESSION_TIMEOUT_ENV, str(DEFAULT_SESSION_TIMEOUT_MINUTES))
    try:
        return max(5, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_SESSION_TIMEOUT_MINUTES


@dataclass
class CheckOutcome:
    check: RealityCheck
    reports: List[Report] = field(default_factory=list)
    safety: Dict[str, Any] = field(default_factory=dict)
    narration_source: str = "offline"

    def public(self) -> Dict[str, Any]:
        return {
            "reality_check": self.check.public(),
            "safety": self.safety,
            "reports": [report.public() for report in self.reports],
            "narration_source": self.narration_source,
        }


class GroundingEngine:
    def __init__(
        self,
        repo: Repository,
        frame_analyzer: Optional[FrameAnalyzer] = None,
        audio_analyzer: Optional[AudioAnalyzer] = None,
        llm: Optional[LLMClient] = None,
        gateway: Any = None,
    ) -> None:
        self.repo = repo
        self.frame_analyzer = frame_analyzer or MetadataFrameAnalyzer()
        self.audio_analyzer = audio_analyzer or MicrophoneAnalyzer()
        self.llm = llm if llm is not None else build_default_llm()
        self.gateway = gateway

    # ------------------------------------------------------------------ main
    def run_reality_check(
        self,
        patient_id: str,
        claim_text: str,
        session_id: Optional[str] = None,
        frame: Optional[FrameSample] = None,
        audio: Optional[AudioSample] = None,
        distress: Optional[int] = None,
    ) -> CheckOutcome:
        patient = self.repo.get_patient(patient_id)
        if not patient:
            raise ValueError("unknown patient")

        session = self.repo.get_session(session_id) if session_id else None
        if session and session.status != "open":
            raise ValueError("this grounding session is already closed")

        # Consent gate: without an open, consented session the device samples are dropped.
        if not session:
            frame, audio = None, None
        else:
            if not session.camera_consent:
                frame = None
            if not session.mic_consent:
                audio = None

        # 1. Deterministic safety screen first.
        risk = safety.screen(claim_text, distress)

        # 2. Structured claim.
        claim: Claim = parse_claim(claim_text)

        # 3. Perception.
        visual: VisualObservation = self.frame_analyzer.analyze(frame)
        sound: AudioObservation = self.audio_analyzer.analyze(audio)

        # 4. Sensor-evidence verdict.
        assessment: Assessment = assess(claim, visual, sound)

        # 5. Risk can only go up: command hallucinations are high risk by definition.
        level = risk.risk_level
        if claim.is_command:
            level = RiskLevel.highest(level, RiskLevel.HIGH)
        if level is not risk.risk_level:
            risk.risk_level = level
            if level is RiskLevel.HIGH and not risk.note:
                risk.note = "Your care team is being notified now."
            if level is RiskLevel.CRITICAL:
                risk.crisis_protocol = True
                risk.note = safety.CRISIS_RESOURCES

        # 6. Narration.
        narration: Narration = narrate(
            claim, assessment.verdict, level, assessment.sensor_summary, self.llm
        )

        # 7. Grounding steps.
        steps = select_steps(claim.modality, assessment.verdict, level, distress or 0)

        escalated = level.rank >= RiskLevel.HIGH.rank

        check = RealityCheck(
            id=new_id("chk"),
            patient_id=patient_id,
            session_id=session_id,
            created_at=utcnow(),
            claim=claim_text.strip() or "(patient asked for a reality check)",
            modality=claim.modality,
            verdict=assessment.verdict,
            confidence=assessment.confidence,
            risk_level=level,
            headline=narration.headline,
            explanation=narration.explanation,
            sensor_summary=f"{assessment.sensor_summary} {assessment.rationale}".strip(),
            grounding_steps=steps,
            entities=claim.entities,
            escalated=escalated,
            disclosure=safety.DISCLOSURE,
        )
        self.repo.add_reality_check(check)
        self.repo.audit(
            patient.user_id,
            "reality_check.created",
            check.id,
            {
                "verdict": check.verdict.value,
                "risk": level.value,
                "session_id": session_id,
                "narration": narration.source,
            },
        )

        reports = self._route_reports(patient, check, risk)

        return CheckOutcome(
            check=check,
            reports=reports,
            safety={**risk.public(), "narration_note": risk.note},
            narration_source=narration.source,
        )

    # --------------------------------------------------------------- reports
    def _route_reports(
        self, patient: Patient, check: RealityCheck, risk: safety.SafetyAssessment
    ) -> List[Report]:
        from ..services.reports import create_report

        summary = (
            f"{check.claim}\n\nSensor evidence ({check.verdict.value}, confidence "
            f"{check.confidence:.0%}): {check.sensor_summary}\n\n"
            f"Risk: {check.risk_level.value}. Flags: {', '.join(risk.flags) or 'none'}."
        )
        payload = check.public()

        reports: List[Report] = [
            create_report(
                self.repo,
                patient_id=patient.id,
                kind="reality_check",
                title=f"Reality check — {check.verdict.value.replace('_', ' ')}",
                summary=summary,
                payload=payload,
                author_id=patient.user_id,
                author_role="patient",
                session_id=check.session_id,
                reality_check_id=check.id,
                urgent=check.escalated,
                gateway=self.gateway,
            )
        ]

        if check.escalated:
            reports.append(
                create_report(
                    self.repo,
                    patient_id=patient.id,
                    kind="crisis_escalation",
                    title=f"Escalation — {check.risk_level.value} risk",
                    summary=(
                        f"Risk level {check.risk_level.value} during a grounding session.\n"
                        f"Patient said: {check.claim}\n"
                        f"Flags: {', '.join(risk.flags) or 'none'}\n"
                        f"{risk.note}"
                    ),
                    payload={**payload, "safety": risk.public()},
                    author_id=patient.user_id,
                    author_role="patient",
                    session_id=check.session_id,
                    reality_check_id=check.id,
                    urgent=True,
                    gateway=self.gateway,
                )
            )
        return reports

    # --------------------------------------------------------------- session
    def open_session(
        self,
        patient_id: str,
        distress: Optional[int] = None,
        camera_consent: bool = False,
        mic_consent: bool = False,
    ) -> Session:
        session = Session(
            id=new_id("ses"),
            patient_id=patient_id,
            started_at=utcnow(),
            distress_start=distress,
            camera_consent=camera_consent,
            mic_consent=mic_consent,
        )
        self.repo.add_session(session)
        return session

    def adopt_session(
        self,
        session: Session,
        distress: Optional[int] = None,
        camera_consent: bool = False,
        mic_consent: bool = False,
    ) -> Session:
        """Reuse an open session, applying the consent choice just made.

        A reload cannot carry browser state, so the patient's latest explicit
        consent (the safe direction: withdrawing wins) is written to the session
        they are continuing instead of opening a duplicate one.
        """
        session.camera_consent = camera_consent
        session.mic_consent = mic_consent
        if session.distress_start is None:
            session.distress_start = distress
        self.repo.update_session(session)
        return session

    def session_activity(self, session: Session) -> str:
        """Timestamp of the last check in this session (or when it opened)."""
        return max(
            [session.started_at]
            + [
                check.created_at
                for check in self.repo.list_reality_checks(session.patient_id)
                if check.session_id == session.id
            ]
        )

    def session_expired(self, session: Session) -> bool:
        if session.status != "open":
            return False
        try:
            last = parse_ts(self.session_activity(session))
        except ValueError:
            return True
        return last < datetime.now(timezone.utc) - timedelta(minutes=session_timeout_minutes())

    def expire_session(self, session: Session) -> Session:
        """Close an idle session, writing the summary the patient never sent."""
        closed = self.close_session(session.id, distress_end=None, reason="timeout")
        self.repo.audit(None, "session.auto_closed", session.id, {"reason": "timeout"})
        return closed

    def resume_session(self, patient_id: str) -> Optional[Session]:
        """The patient's live session, if any. Idle ones are closed first."""
        live: List[Session] = []
        for session in self.repo.list_sessions(patient_id):
            if session.status != "open":
                continue
            if self.session_expired(session):
                self.expire_session(session)
            else:
                live.append(session)
        live.sort(key=lambda item: item.started_at)
        return live[-1] if live else None

    def close_session(
        self,
        session_id: str,
        distress_end: Optional[int] = None,
        reason: str = "patient",
    ) -> Session:
        session = self.repo.get_session(session_id)
        if not session:
            raise ValueError("unknown session")
        if session.status != "open":
            return session
        session.ended_at = utcnow()
        session.distress_end = distress_end
        session.status = "closed"
        self.repo.update_session(session)

        from ..services.reports import create_report

        checks = [
            check
            for check in self.repo.list_reality_checks(session.patient_id)
            if check.session_id == session.id
        ]
        relief = None
        if session.distress_start is not None and distress_end is not None:
            relief = session.distress_start - distress_end
        worst = max((check.risk_level for check in checks), key=lambda lvl: lvl.rank, default=RiskLevel.LOW)
        if session.distress_start is None:
            distress = "No distress rating at the start."
        elif distress_end is None:
            distress = f"Distress at the start {session.distress_start}; no end rating."
        else:
            distress = f"Distress {session.distress_start} → {distress_end}"
            if relief is not None:
                distress += f" (relief {relief})"
            distress += "."
        closed_by = (
            f" Closed automatically after {session_timeout_minutes()} minutes without activity."
            if reason == "timeout"
            else ""
        )
        summary = (
            f"Session closed after {len(checks)} reality check(s). {distress}"
            f" Peak risk: {worst.value}.{closed_by}"
        )
        patient = self.repo.get_patient(session.patient_id)
        if patient:
            create_report(
                self.repo,
                patient_id=patient.id,
                kind="session_summary",
                title="Grounding session summary",
                summary=summary,
                payload={
                    "session": session.public(),
                    "checks": [check.public() for check in checks],
                    "relief": relief,
                    "closed_reason": reason,
                },
                author_id=patient.user_id,
                author_role="patient",
                session_id=session.id,
                urgent=worst.rank >= RiskLevel.HIGH.rank,
                gateway=self.gateway,
            )
        return session
