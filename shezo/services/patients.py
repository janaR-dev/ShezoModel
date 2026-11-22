"""Care-team and patient provisioning helpers."""

from __future__ import annotations

from typing import Optional

from ..models import CareTeamMember, Patient, Role, User, utcnow
from ..storage import Repository, new_id


def create_patient(
    repo: Repository,
    user: User,
    display_name: str,
    therapist_id: Optional[str] = None,
    primary_caretaker_id: Optional[str] = None,
    care_notes: str = "",
    emergency_contact: str = "",
    date_of_birth: Optional[str] = None,
) -> Patient:
    if user.role is not Role.PATIENT:
        raise ValueError("the linked user must have the patient role")
    if repo.patient_for_user(user.id):
        raise ValueError("this user already has a patient record")
    patient = Patient(
        id=new_id("pat"),
        user_id=user.id,
        display_name=display_name or user.display_name,
        created_at=utcnow(),
        therapist_id=therapist_id,
        primary_caretaker_id=primary_caretaker_id,
        care_notes=care_notes,
        emergency_contact=emergency_contact,
        date_of_birth=date_of_birth,
    )
    repo.add_patient(patient)
    if therapist_id:
        assign_member(repo, patient.id, therapist_id, Role.THERAPIST, "therapist")
    if primary_caretaker_id:
        assign_member(repo, patient.id, primary_caretaker_id, Role.CARETAKER, "primary caretaker")
    return patient


def assign_member(
    repo: Repository,
    patient_id: str,
    user_id: str,
    role: Role,
    relationship: str = "",
) -> CareTeamMember:
    user = repo.get_user(user_id)
    if not user:
        raise ValueError("unknown care team member")
    if user.role is not role:
        raise ValueError(f"{user.display_name} is not registered as a {role.value}")
    member = CareTeamMember(
        patient_id=patient_id,
        user_id=user_id,
        role=role,
        relationship=relationship,
        display_name=user.display_name,
    )
    repo.add_care_member(member)
    repo.audit(user_id, "care_team.assigned", patient_id, {"role": role.value})
    return member
