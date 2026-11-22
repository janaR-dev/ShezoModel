"""Medication reminders and adherence.

Times are stored as local wall-clock "HH:MM" strings plus a timezone offset. The
scheduler materialises one ``Dose`` row per due time per day, so adherence is a
query rather than a guess. A dose still pending after the grace window becomes
``missed`` once, and that transition produces exactly one caretaker alert.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..models import Dose, DoseStatus, ReportKind, Role, parse_ts, utcnow
from ..storage import Repository, new_id
from .reports import create_report

DEFAULT_GRACE_MINUTES = 60


def _parse_time(text: str) -> "tuple[int, int]":
    cleaned = (text or "").strip()
    if "T" in cleaned:
        cleaned = cleaned.split("T")[-1]
    parts = cleaned.split(":")
    if len(parts) < 2:
        raise ValueError(f"invalid medication time: {text!r} (expected HH:MM)")
    hour, minute = int(parts[0]), int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"invalid medication time: {text!r}")
    return hour, minute


def _due_at_iso(day: datetime, hhmm: str, tz_offset_minutes: int) -> str:
    hour, minute = _parse_time(hhmm)
    local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)
    return (local - timedelta(minutes=tz_offset_minutes)).replace(microsecond=0).isoformat()


def local_day(now: str, tz_offset_minutes: int) -> datetime:
    return (parse_ts(now) + timedelta(minutes=tz_offset_minutes)).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    )


def ensure_doses(
    repo: Repository,
    patient_id: str,
    day: Optional[datetime] = None,
    now: Optional[str] = None,
    tz_offset_minutes: int = 0,
) -> List[Dose]:
    """Idempotently create today's dose rows for every active medication."""
    now = now or utcnow()
    day = day or local_day(now, tz_offset_minutes)
    created: List[Dose] = []
    for med in repo.list_medications(patient_id, active_only=True):
        for hhmm in med.times:
            due_at = _due_at_iso(day, hhmm, tz_offset_minutes)
            if repo.list_doses(patient_id, due_from=due_at, due_to=due_at):
                continue
            dose = Dose(
                id=new_id("dose"),
                medication_id=med.id,
                patient_id=patient_id,
                due_at=due_at,
                status=DoseStatus.PENDING,
                medication_name=med.name,
                medication_dose=med.dose,
            )
            repo.upsert_dose(dose)
            created.append(dose)
    return created


def refresh_missed(
    repo: Repository,
    patient_id: str,
    now: Optional[str] = None,
    grace_minutes: int = DEFAULT_GRACE_MINUTES,
    gateway: Any = None,
) -> List[Dose]:
    """Promote overdue pending doses to missed and alert the care team once."""
    now = now or utcnow()
    cutoff = (parse_ts(now) - timedelta(minutes=grace_minutes)).replace(microsecond=0).isoformat()
    overdue = repo.list_doses(patient_id, due_to=cutoff, statuses=[DoseStatus.PENDING])

    escalated: List[Dose] = []
    for dose in overdue:
        repo.set_dose_status(dose.id, DoseStatus.MISSED, None)
        dose.status = DoseStatus.MISSED
        escalated.append(dose)

    if escalated:
        listing = "; ".join(
            f"{dose.medication_name} {dose.medication_dose} due {dose.due_at}" for dose in escalated
        )
        create_report(
            repo,
            patient_id=patient_id,
            kind=ReportKind.MEDICATION_MISSED,
            title=f"Missed medication — {len(escalated)} dose(s)",
            summary=f"No confirmation of these doses after {grace_minutes} minutes: {listing}",
            payload={"doses": [dose.public() for dose in escalated]},
            author_id="system",
            author_role=Role.CARETAKER,
            urgent=len(escalated) > 1,
            gateway=gateway,
        )
        repo.audit(None, "medication.missed", patient_id, {"dose_ids": [d.id for d in escalated]})
    return escalated


def mark_dose(
    repo: Repository,
    dose_id: str,
    action: str,
    actor_id: str,
    now: Optional[str] = None,
) -> Dose:
    dose = repo.get_dose(dose_id)
    if not dose:
        raise ValueError("unknown dose")
    normalised = (action or "").lower()
    if normalised in ("taken", "take", "took"):
        status = DoseStatus.TAKEN
    elif normalised in ("skipped", "skip", "declined"):
        status = DoseStatus.SKIPPED
    else:
        raise ValueError("action must be 'taken' or 'skipped'")
    if dose.status in (DoseStatus.TAKEN, DoseStatus.SKIPPED):
        raise ValueError(f"this dose is already marked {dose.status.value}")
    repo.set_dose_status(dose.id, status, now or utcnow())
    repo.audit(actor_id, f"medication.{status.value}", dose_id)
    updated = repo.get_dose(dose_id)
    assert updated is not None
    return updated


def due_reminders(
    repo: Repository,
    patient_id: str,
    now: Optional[str] = None,
    window_minutes: int = 45,
    lookahead_minutes: int = 120,
    tz_offset_minutes: int = 0,
    grace_minutes: int = DEFAULT_GRACE_MINUTES,
) -> Dict[str, Any]:
    """What the patient's reminder card should show right now."""
    now = now or utcnow()
    ensure_doses(repo, patient_id, now=now, tz_offset_minutes=tz_offset_minutes)
    refresh_missed(repo, patient_id, now=now, grace_minutes=grace_minutes)

    current = parse_ts(now)
    window_start = (current - timedelta(minutes=window_minutes)).isoformat()
    window_end = (current + timedelta(minutes=window_minutes)).isoformat()
    lookahead = (current + timedelta(minutes=lookahead_minutes)).isoformat()

    taken = repo.list_doses(
        patient_id, due_from=window_start, due_to=window_end, statuses=[DoseStatus.TAKEN]
    )
    pending = repo.list_doses(
        patient_id, due_from=window_start, due_to=window_end, statuses=[DoseStatus.PENDING]
    )
    # Missed doses stay visible for 12 hours, not just the reminder window, so the
    # patient can still confirm or explain a dose they took late.
    missed = repo.list_doses(
        patient_id,
        due_from=(current - timedelta(hours=12)).isoformat(),
        due_to=window_end,
        statuses=[DoseStatus.MISSED],
    )
    upcoming = repo.list_doses(
        patient_id, due_from=window_end, due_to=lookahead, statuses=[DoseStatus.PENDING]
    )

    return {
        "generated_at": now,
        "due_now": [dose.public() for dose in pending],
        "recently_taken": [dose.public() for dose in taken],
        "missed": [dose.public() for dose in missed],
        "upcoming": [dose.public() for dose in upcoming],
        "message": _reminder_message(pending, upcoming, missed),
    }


def _reminder_message(pending: List[Dose], upcoming: List[Dose], missed: List[Dose]) -> str:
    if missed and pending:
        first = pending[0]
        return (
            f"{len(missed)} earlier dose(s) were not confirmed, and {first.medication_name} "
            f"{first.medication_dose} is due now. Tell your care team about the missed dose."
        )
    if missed:
        return (
            f"{len(missed)} dose(s) were not confirmed. Take them now if your prescriber said it is safe, "
            "and tell your care team."
        )
    if pending:
        first = pending[0]
        return f"Time for {first.medication_name} {first.medication_dose}. Tap taken when you have it."
    if upcoming:
        first = upcoming[0]
        return f"Next: {first.medication_name} {first.medication_dose} at {first.due_at[11:16]} UTC."
    return "Nothing due right now. Nice work keeping the routine going."


def adherence(repo: Repository, patient_id: str, since: Optional[str] = None) -> Dict[str, Any]:
    doses = repo.list_doses(patient_id, due_from=since)
    counts = {status.value: 0 for status in DoseStatus}
    for dose in doses:
        counts[dose.status.value] += 1
    settled = counts["taken"] + counts["skipped"] + counts["missed"]
    pct = round(100 * counts["taken"] / settled, 1) if settled else None
    return {**counts, "settled": settled, "adherence_pct": pct}
