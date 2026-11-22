"""Report creation, routing to the care team, and inbox handling.

Every reality check produces one report addressed to the patient's therapist and
all assigned caretakers. High and critical risk additionally produce an urgent
escalation report. Routing is derived from the care team on the server; a client
can never choose its own recipients.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import Report, ReportKind, Role, User, utcnow
from ..storage import Repository, new_id
from .notifications import build_gateway

DEFAULT_ROLES = (Role.THERAPIST, Role.CARETAKER)


def care_recipients(repo: Repository, patient_id: str) -> List[User]:
    """Therapist plus every caretaker assigned in the care team (deduplicated)."""
    patient = repo.get_patient(patient_id)
    if not patient:
        return []
    user_ids: List[str] = []
    for member in repo.care_team(patient_id):
        if member.role in DEFAULT_ROLES:
            user_ids.append(member.user_id)
    if patient.therapist_id:
        user_ids.append(patient.therapist_id)
    if patient.primary_caretaker_id:
        user_ids.append(patient.primary_caretaker_id)

    recipients: List[User] = []
    seen = set()
    for user_id in user_ids:
        if user_id in seen:
            continue
        seen.add(user_id)
        user = repo.get_user(user_id)
        if user and user.role in DEFAULT_ROLES:
            recipients.append(user)
    return recipients


def create_report(
    repo: Repository,
    *,
    patient_id: str,
    kind: "str | ReportKind",
    title: str,
    summary: str,
    payload: Optional[Dict[str, Any]] = None,
    author_id: str = "",
    author_role: "str | Role" = Role.PATIENT,
    session_id: Optional[str] = None,
    reality_check_id: Optional[str] = None,
    urgent: bool = False,
    gateway: Any = None,
    recipients: Optional[List[User]] = None,
) -> Report:
    kind_enum = kind if isinstance(kind, ReportKind) else ReportKind(str(kind))
    role_enum = author_role if isinstance(author_role, Role) else Role(str(author_role))

    report = Report(
        id=new_id("rep"),
        patient_id=patient_id,
        kind=kind_enum,
        title=title,
        summary=summary,
        author_role=role_enum,
        author_id=author_id,
        created_at=utcnow(),
        payload=dict(payload or {}),
        session_id=session_id,
        reality_check_id=reality_check_id,
        urgent=urgent,
    )
    repo.add_report(report)

    targets = recipients if recipients is not None else care_recipients(repo, patient_id)
    if not targets:
        repo.add_delivery(report.id, "", "inbox", "unrouted", "no therapist or caretaker assigned")
        report.payload = {**report.payload, "routing": "unrouted"}
        report.recipients = []
        repo.audit(None, "report.created_untargeted", report.id, {"kind": kind_enum.value})
        return report

    gateway = gateway or build_gateway()
    detailed: List[Dict[str, Any]] = []
    for recipient in targets:
        repo.add_recipient(report.id, recipient.id, recipient.role)
        try:
            status, detail = gateway.send(report, recipient)
        except Exception as exc:  # noqa: BLE001 - one bad channel must not lose the report
            status, detail = "failed", f"{type(exc).__name__}: {exc}"
        channel = getattr(gateway, "channel", "inbox")
        repo.add_delivery(report.id, recipient.id, channel, status, detail)
        if status in ("sent", "recorded"):
            repo.mark_delivered(report.id, recipient.id)
        detailed.append(
            {
                "user_id": recipient.id,
                "role": recipient.role.value,
                "name": recipient.display_name,
                "channel": channel,
                "status": status,
            }
        )

    report.recipients = detailed
    repo.audit(
        author_id or None,
        "report.routed",
        report.id,
        {"kind": kind_enum.value, "urgent": urgent, "recipients": [d["user_id"] for d in detailed]},
    )
    return report


def inbox(repo: Repository, user_id: str, unread_only: bool = False) -> List[Dict[str, Any]]:
    reports = repo.list_reports_for_user(user_id, unread_only=unread_only)
    for report in reports:
        repo.mark_read(report.id, user_id)
    return [report.public() for report in reports]


def acknowledge(repo: Repository, report_id: str, user_id: str) -> bool:
    report = repo.get_report(report_id)
    if not report:
        raise ValueError("unknown report")
    if user_id not in repo.report_recipient_ids(report_id):
        raise PermissionError("this report was not addressed to you")
    ok = repo.acknowledge(report_id, user_id)
    repo.audit(user_id, "report.acknowledged", report_id)
    return ok
