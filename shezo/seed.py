"""Seed the configured database with demo accounts and two weeks of history.

    py -m shezo.seed                 # MySQL by default (see SHEZO_DB_ENGINE)
    SHEZO_DB_ENGINE=sqlite py -m shezo.seed

Everything written here is demo content for the three web dashboards. It is
created through the same repository and model helpers the app uses, so the
records look like real ones (real narrations, real grounding steps, real report
routing) rather than hand-written fixtures.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .ai.claims import parse_claim
from .ai.grounding import select_steps
from .ai.llm import narrate
from .ai.safety import DISCLOSURE
from .auth import create_user
from .models import (
    DoseStatus,
    Medication,
    Modality,
    RealityCheck,
    Report,
    ReportKind,
    RiskLevel,
    Role,
    Session,
    User,
    Verdict,
    utcnow,
)
from .services.medications import ensure_doses
from .services.patients import create_patient
from .storage import Repository, new_id, open_backend

PATIENT_ONE = {
    "email": "jordan@example.com",
    "password": "patient-pass-1",
    "name": "Jordan",
}
PATIENT_TWO = {
    "email": "aisha@example.com",
    "password": "patient-pass-2",
    "name": "Aisha",
}
THERAPIST = {
    "email": "dr.reyes@clinic.example",
    "password": "therapy-pass-1",
    "name": "Dr. Amara Reyes",
}
CARETAKER = {
    "email": "sam@family.example",
    "password": "family-pass-1",
    "name": "Sam Okafor",
}

_SENSOR_TEXT = {
    Verdict.CORROBORATED: "Camera saw: person (confidence up to 93%). Microphone recorded speech during the check.",
    Verdict.NO_SENSOR_EVIDENCE: (
        "Camera was on and the frame was usable, but the detector recognised no person or object of interest. "
        "Microphone recorded no speech, only room tone around 31 dB."
    ),
    Verdict.INCONCLUSIVE: (
        "Camera image was too dark or too blurry to tell us anything. Microphone was off for this check."
    ),
}


def _iso(day: datetime, hour: int, minute: int = 0) -> str:
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0, tzinfo=timezone.utc).isoformat()


def _add_check(
    repo: Repository,
    patient_id: str,
    text: str,
    verdict: Verdict,
    risk: RiskLevel,
    distress: int,
    created_at: str,
    session_id: Optional[str] = None,
) -> RealityCheck:
    claim = parse_claim(text)
    summary = _SENSOR_TEXT[verdict]
    narration = narrate(claim, verdict, risk, summary, None)
    modality = claim.modality if claim.modality is not Modality.OTHER else Modality.VISUAL
    check = RealityCheck(
        id=new_id("chk"),
        patient_id=patient_id,
        session_id=session_id,
        created_at=created_at,
        claim=text,
        modality=modality,
        verdict=verdict,
        confidence={Verdict.CORROBORATED: 0.86, Verdict.NO_SENSOR_EVIDENCE: 0.71, Verdict.INCONCLUSIVE: 0.28}[verdict],
        risk_level=risk,
        headline=narration.headline,
        explanation=narration.explanation,
        sensor_summary=summary,
        grounding_steps=select_steps(modality, verdict, risk, distress),
        entities=claim.entities,
        escalated=risk.rank >= RiskLevel.HIGH.rank,
        disclosure=DISCLOSURE,
    )
    repo.add_reality_check(check)
    return check


def _add_report(
    repo: Repository,
    patient_id: str,
    check: RealityCheck,
    kind: ReportKind,
    title: str,
    summary: str,
    created_at: str,
    urgent: bool = False,
    acknowledge_for: Optional[User] = None,
) -> Report:
    report = Report(
        id=new_id("rep"),
        patient_id=patient_id,
        kind=kind,
        title=title,
        summary=summary,
        author_role=Role.PATIENT,
        author_id="",
        created_at=created_at,
        payload=check.public(),
        session_id=check.session_id,
        reality_check_id=check.id,
        urgent=urgent,
    )
    repo.add_report(report)
    for member in repo.care_team(patient_id):
        repo.add_recipient(report.id, member.user_id, member.role)
        repo.add_delivery(report.id, member.user_id, "inbox", "recorded", "seeded demo delivery")
        repo.mark_delivered(report.id, member.user_id, created_at)
        if acknowledge_for is not None and member.user_id == acknowledge_for.id:
            repo.acknowledge(report.id, member.user_id)
    return report


def _history(
    repo: Repository,
    patient_id: str,
    days: int,
    start_distress: int,
    end_distress: int,
    acknowledge_for: Optional[User] = None,
) -> None:
    """Create one closed grounding session and 1-2 checks per day."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    verdict_cycle = [
        (Verdict.NO_SENSOR_EVIDENCE, RiskLevel.LOW, "There is a man standing in the corner by the door watching me."),
        (Verdict.INCONCLUSIVE, RiskLevel.LOW, "The light in the hallway keeps flashing. Is this real?"),
        (Verdict.NO_SENSOR_EVIDENCE, RiskLevel.ELEVATED, "I can hear whispering from the wall again."),
        (Verdict.CORROBORATED, RiskLevel.LOW, "Is someone at the window? I think I see a person."),
        (Verdict.NO_SENSOR_EVIDENCE, RiskLevel.LOW, "There are shadows moving under the door."),
    ]

    for index in range(days, 0, -1):
        day = now - timedelta(days=index)
        progress = (days - index) / max(days - 1, 1)
        opening = round(start_distress - (start_distress - end_distress) * progress)
        closing = max(1, opening - 3)
        session = Session(
            id=new_id("ses"),
            patient_id=patient_id,
            started_at=_iso(day, 19, 10),
            ended_at=_iso(day, 19, 34),
            distress_start=opening,
            distress_end=closing,
            camera_consent=True,
            mic_consent=index % 3 != 0,
            status="closed",
        )
        repo.add_session(session)

        verdict, risk, text = verdict_cycle[index % len(verdict_cycle)]
        if index == days - 2:
            verdict, risk, text = (
                Verdict.NO_SENSOR_EVIDENCE,
                RiskLevel.CRITICAL,
                "The voices are telling me to go outside and meet them.",
            )
        check = _add_check(
            repo,
            patient_id,
            text,
            verdict,
            risk,
            opening,
            _iso(day, 19, 16),
            session_id=session.id,
        )
        _add_report(
            repo,
            patient_id,
            check,
            ReportKind.REALITY_CHECK,
            f"Reality check — {verdict.value.replace('_', ' ')}",
            f"{text}\n\nSensor evidence ({verdict.value}, confidence {check.confidence:.0%}): {check.sensor_summary}\n\nRisk: {risk.value}.",
            _iso(day, 19, 16),
            urgent=check.escalated,
            acknowledge_for=acknowledge_for if index > 3 else None,
        )
        if check.escalated:
            crisis = _add_check(
                repo,
                patient_id,
                text,
                verdict,
                risk,
                opening,
                _iso(day, 19, 17),
                session_id=session.id,
            )
            _add_report(
                repo,
                patient_id,
                crisis,
                ReportKind.CRISIS_ESCALATION,
                f"Escalation — {risk.value} risk",
                f"Risk level {risk.value} during a grounding session.\nPatient said: {text}",
                _iso(day, 19, 17),
                urgent=True,
            )
        if index % 2 == 0:
            extra = _add_check(
                repo,
                patient_id,
                "The voices are quieter tonight, but I keep checking the hallway.",
                Verdict.NO_SENSOR_EVIDENCE,
                RiskLevel.LOW,
                max(1, opening - 2),
                _iso(day, 21, 40),
                session_id=session.id,
            )
            _add_report(
                repo,
                patient_id,
                extra,
                ReportKind.REALITY_CHECK,
                "Reality check — no sensor evidence",
                f"{extra.claim}\n\nRisk: low. No escalation.",
                _iso(day, 21, 40),
            )


def _medication_history(repo: Repository, patient_id: str, days: int) -> None:
    """Back-fill doses: mostly taken, with a couple of missed evenings."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    for index in range(days, -1, -1):
        day = (now - timedelta(days=index)).replace(hour=0, minute=0, second=0, microsecond=0)
        ensure_doses(repo, patient_id, day=day)
        doses = repo.list_doses(
            patient_id,
            due_from=day.isoformat(),
            due_to=(day + timedelta(days=1)).isoformat(),
        )
        for dose in doses:
            if dose.status is not DoseStatus.PENDING:
                continue
            if dose.due_at > now.isoformat():
                continue  # still in the future: leave it pending
            hour = int(dose.due_at[11:13])
            missed = index in (4, 9) and hour >= 12
            skipped = index == 6 and hour >= 12
            if missed:
                repo.set_dose_status(dose.id, DoseStatus.MISSED, None)
            elif skipped:
                repo.set_dose_status(dose.id, DoseStatus.SKIPPED, dose.due_at)
            else:
                repo.set_dose_status(dose.id, DoseStatus.TAKEN, dose.due_at)


def seed(repo: Repository) -> Dict[str, User]:
    therapist = create_user(repo, THERAPIST["email"], THERAPIST["password"], Role.THERAPIST, THERAPIST["name"])
    caretaker = create_user(repo, CARETAKER["email"], CARETAKER["password"], Role.CARETAKER, CARETAKER["name"])
    jordan = create_user(repo, PATIENT_ONE["email"], PATIENT_ONE["password"], Role.PATIENT, PATIENT_ONE["name"])
    aisha = create_user(repo, PATIENT_TWO["email"], PATIENT_TWO["password"], Role.PATIENT, PATIENT_TWO["name"])

    patient = create_patient(
        repo,
        jordan,
        "Jordan",
        therapist_id=therapist.id,
        primary_caretaker_id=caretaker.id,
        care_notes="Schizophrenia, on risperidone. Prefers being asked before the camera is used.",
        emergency_contact="Sam Okafor +1 555 0100",
    )
    repo.add_medication(
        Medication(
            id=new_id("med"),
            patient_id=patient.id,
            name="Risperidone",
            dose="2 mg",
            times=["09:00", "21:00"],
            instructions="With food.",
            created_at=utcnow(),
        )
    )
    soon = (datetime.now(timezone.utc) + timedelta(minutes=3)).strftime("%H:%M")
    repo.add_medication(
        Medication(
            id=new_id("med"),
            patient_id=patient.id,
            name="Propranolol",
            dose="10 mg",
            times=[soon],
            instructions="Only if your prescriber approved it for acute anxiety.",
            created_at=utcnow(),
        )
    )

    second = create_patient(
        repo,
        aisha,
        "Aisha",
        therapist_id=therapist.id,
        primary_caretaker_id=caretaker.id,
        care_notes="First-episode psychosis, stepped down from inpatient care last month.",
        emergency_contact="Sam Okafor +1 555 0100",
    )
    repo.add_medication(
        Medication(
            id=new_id("med"),
            patient_id=second.id,
            name="Aripiprazole",
            dose="5 mg",
            times=["08:30"],
            instructions="Morning, with water.",
            created_at=utcnow(),
        )
    )

    _history(repo, patient.id, days=24, start_distress=9, end_distress=4, acknowledge_for=caretaker)
    _history(repo, second.id, days=12, start_distress=7, end_distress=5)
    _medication_history(repo, patient.id, days=24)
    _medication_history(repo, second.id, days=12)
    ensure_doses(repo, patient.id)

    repo.audit(therapist.id, "demo.seeded", patient.id, {"patients": 2})
    return {"therapist": therapist, "caretaker": caretaker, "patient": jordan, "patient_two": aisha}


def main() -> None:
    repo, label = open_backend()
    try:
        if repo.get_user_by_email(THERAPIST["email"]):
            print(f"Demo data already present in {label}; nothing to do.")
            return
        seed(repo)
        print(f"Seeded {label}")
        print("\nSign in at http://127.0.0.1:8080/")
        for account, role in (
            (PATIENT_ONE, "patient"),
            (PATIENT_TWO, "patient"),
            (CARETAKER, "caretaker"),
            (THERAPIST, "therapist"),
        ):
            print(f"  {role:<9} {account['email']:<28} {account['password']}")
    finally:
        repo.close()


if __name__ == "__main__":
    main()
