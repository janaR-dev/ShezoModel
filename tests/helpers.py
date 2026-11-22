"""Shared fixtures for the test suite."""
from __future__ import annotations

from typing import Any, Dict

from shezo.ai.engine import GroundingEngine
from shezo.auth import create_user
from shezo.models import Medication, Role, utcnow
from shezo.services.patients import create_patient
from shezo.storage import Repository
from shezo.api import AppContext, AuthService
from shezo.services.notifications import RecordingGateway

PASSWORDS = {
    "therapist": "therapist-pass-1",
    "caretaker": "caretaker-pass-1",
    "patient": "patient-pass-1",
}


def build_world(db_path: str = ":memory:") -> Dict[str, Any]:
    repo = Repository(db_path)
    gateway = RecordingGateway()
    ctx = AppContext(
        repo=repo,
        auth=AuthService(repo),
        engine=GroundingEngine(repo, gateway=gateway, llm=None),
        gateway=gateway,
    )

    therapist = create_user(repo, "t@clinic.example", PASSWORDS["therapist"], Role.THERAPIST, "Dr. Test")
    caretaker = create_user(repo, "c@home.example", PASSWORDS["caretaker"], Role.CARETAKER, "Casey Test")
    patient_user = create_user(repo, "p@home.example", PASSWORDS["patient"], Role.PATIENT, "Pat Test")
    patient = create_patient(
        repo,
        patient_user,
        "Pat Test",
        therapist_id=therapist.id,
        primary_caretaker_id=caretaker.id,
        care_notes="Test record.",
        emergency_contact="Casey Test",
    )
    repo.add_medication(
        Medication(
            id="med_test",
            patient_id=patient.id,
            name="Risperidone",
            dose="2 mg",
            times=["09:00", "21:00"],
            created_at=utcnow(),
        )
    )

    return {
        "repo": repo,
        "ctx": ctx,
        "engine": ctx.engine,
        "therapist": therapist,
        "caretaker": caretaker,
        "patient_user": patient_user,
        "patient": patient,
        "passwords": dict(PASSWORDS),
    }
