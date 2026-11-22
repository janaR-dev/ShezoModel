"""End-to-end demo of the model layer, no server and no UI.

    py -m shezo.demo
    py -m shezo.demo --db shezo_demo.db

Seeds a therapist, a caretaker and a patient, then plays a full session: a visual
report the camera cannot confirm, an auditory report, a command hallucination
that escalates, a medication reminder that is taken, and one that is missed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from .ai.engine import GroundingEngine
from .ai.perception import AudioSample, FrameSample
from .ai.safety import screen
from .auth import create_user
from .models import Medication, Role, utcnow
from .services import medications, progress
from .services.patients import create_patient
from .services.reports import inbox
from .storage import Repository


def _p(title: str, payload: Any) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(json.dumps(payload, indent=2, default=str)[:4000])


def seed(repo: Repository) -> Dict[str, Any]:
    therapist = create_user(repo, "dr.reyes@clinic.example", "therapy-pass-1", Role.THERAPIST, "Dr. Amara Reyes")
    caretaker = create_user(repo, "sam@family.example", "family-pass-1", Role.CARETAKER, "Sam Okafor")
    patient_user = create_user(repo, "jordan@example.com", "patient-pass-1", Role.PATIENT, "Jordan")

    patient = create_patient(
        repo,
        patient_user,
        "Jordan",
        therapist_id=therapist.id,
        primary_caretaker_id=caretaker.id,
        care_notes="Schizophrenia, on risperidone. Prefers being asked before the camera is used.",
        emergency_contact="Sam Okafor +1 555 0100",
    )
    repo.add_medication(
        Medication(
            id="med_demo_risperidone",
            patient_id=patient.id,
            name="Risperidone",
            dose="2 mg",
            times=["09:00", "21:00"],
            instructions="With food.",
            created_at=utcnow(),
        )
    )
    # A dose that is due right now, so the reminder card shows real content.
    soon = (datetime.now(timezone.utc) + timedelta(minutes=2)).strftime("%H:%M")
    repo.add_medication(
        Medication(
            id="med_demo_prn",
            patient_id=patient.id,
            name="Propranolol",
            dose="10 mg",
            times=[soon],
            instructions="Take only if your prescriber approved it for acute anxiety.",
            created_at=utcnow(),
        )
    )
    return {
        "therapist": therapist,
        "caretaker": caretaker,
        "patient_user": patient_user,
        "patient": patient,
    }


def run(db_path: str) -> None:
    repo = Repository(db_path)
    world = seed(repo)
    engine = GroundingEngine(repo)
    patient = world["patient"]

    _p(
        "1. CARE TEAM (one shared record, three interfaces)",
        {
            "patient": patient.public(),
            "care_team": [member.__dict__ for member in repo.care_team(patient.id)],
        },
    )

    session = engine.open_session(patient.id, distress=8, camera_consent=True, mic_consent=True)
    _p("2. SESSION OPENED (consent gates the camera/mic)", session.public())

    # Camera on, nothing in the room.
    visual = engine.run_reality_check(
        patient.id,
        "There is a man standing in the corner by the door watching me.",
        session_id=session.id,
        frame=FrameSample(labels={"chair": 0.62, "door": 0.88}, brightness=0.55, blur=0.1),
        audio=AudioSample(duration_s=3.0, ambient_db=31.0, speech_detected=False),
        distress=8,
    )
    _p(
        "3. VISUAL REPORT — camera on, nothing matching detected",
        {
            "verdict": visual.check.verdict.value,
            "headline": visual.check.headline,
            "explanation": visual.check.explanation,
            "sensor_summary": visual.check.sensor_summary,
            "grounding_step_titles": [step["title"] for step in visual.check.grounding_steps],
            "routed_to": visual.reports[0].recipients,
        },
    )

    # Camera actually sees a person: the model says something is there.
    corroborated = engine.run_reality_check(
        patient.id,
        "Is someone at the window? I think I see a person.",
        session_id=session.id,
        frame=FrameSample(labels={"person": 0.93, "window": 0.81}, brightness=0.7, blur=0.05),
        audio=AudioSample(duration_s=3.0, speech_detected=True, other_speaker=True, ambient_db=48.0),
        distress=6,
    )
    _p(
        "4. CORROBORATED REPORT — camera did detect a person",
        {
            "verdict": corroborated.check.verdict.value,
            "headline": corroborated.check.headline,
            "explanation": corroborated.check.explanation,
        },
    )

    # Auditory: mic records no second voice.
    auditory = engine.run_reality_check(
        patient.id,
        "The voices are telling me to go outside and meet them.",
        session_id=session.id,
        frame=FrameSample(labels={}, brightness=0.5, blur=0.05),
        audio=AudioSample(duration_s=3.0, ambient_db=29.0, speech_detected=False),
        distress=9,
    )
    _p(
        "5. COMMAND HALLUCINATION — escalates to the care team",
        {
            "verdict": auditory.check.verdict.value,
            "risk": auditory.check.risk_level.value,
            "flags": auditory.safety["flags"],
            "escalated": auditory.check.escalated,
            "explanation": auditory.check.explanation,
            "reports": [
                {"kind": report.kind.value, "urgent": report.urgent, "recipients": len(report.recipients)}
                for report in auditory.reports
            ],
            "steps": [step["title"] for step in auditory.check.grounding_steps],
        },
    )

    engine.close_session(session.id, distress_end=4)

    # Medication: take the morning dose, leave the evening one to be missed.
    now = datetime.now(timezone.utc).replace(microsecond=0)
    morning = (now - timedelta(hours=2)).replace(minute=0, second=0).isoformat()
    medications.ensure_doses(repo, patient.id, now=morning)
    doses = [
        dose
        for dose in repo.list_doses(patient.id, due_from=(now - timedelta(hours=6)).isoformat())
        if dose.medication_name == "Risperidone"
    ]
    if doses:
        medications.mark_dose(repo, doses[0].id, "taken", world["patient_user"].id)
    medications.refresh_missed(repo, patient.id, now=utcnow())

    _p("6. MEDICATION REMINDERS", medications.due_reminders(repo, patient.id))

    _p(
        "7. CARETAKER VIEW — improvement overview",
        progress.caretaker_dashboard(repo, world["caretaker"].id),
    )

    _p(
        "8. THERAPIST VIEW — progress + reports",
        {
            "progress": progress.patient_progress(repo, patient.id),
            "inbox": inbox(repo, world["therapist"].id)[:2],
        },
    )

    _p(
        "9. SAFETY FLOOR (deterministic, runs before any model)",
        {
            "plain": screen("I feel shaky but I know you are checking with me.").public(),
            "crisis": screen("I want to kill myself tonight.").public(),
        },
    )

    print("\nDemo complete. This is decision support, not a medical device.")
    repo.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Shezo model demo")
    parser.add_argument("--db", default=":memory:", help="SQLite path (default: in-memory)")
    args = parser.parse_args()
    run(args.db)


if __name__ == "__main__":
    main()
