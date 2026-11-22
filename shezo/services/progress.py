"""Progress aggregation for the caretaker and therapist views.

Everything is computed from the shared record: reality checks, sessions, doses,
reports and notes. No separate "dashboard database", so the three interfaces can
never disagree about what happened.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..models import DoseStatus, ReportKind, RiskLevel, Role, Verdict, parse_ts, utcnow
from ..storage import Repository
from .medications import adherence, ensure_doses, refresh_missed
from .reports import inbox


def _avg(values: List[float]) -> Optional[float]:
    cleaned = [v for v in values if v is not None]
    return round(sum(cleaned) / len(cleaned), 2) if cleaned else None


def _bucket_start(ts: str, days: int = 7) -> str:
    day = parse_ts(ts).replace(hour=0, minute=0, second=0, microsecond=0)
    start = day - timedelta(days=day.weekday())
    return start.isoformat()


def patient_progress(
    repo: Repository,
    patient_id: str,
    days: int = 14,
    now: Optional[str] = None,
) -> Dict[str, Any]:
    now = now or utcnow()
    current_end = parse_ts(now)
    window_start = (current_end - timedelta(days=days)).isoformat()
    prev_start = (current_end - timedelta(days=days * 2)).isoformat()

    sessions = repo.list_sessions(patient_id, since=window_start)
    checks = repo.list_reality_checks(patient_id, since=window_start)
    prev_checks = [
        check
        for check in repo.list_reality_checks(patient_id, since=prev_start)
        if check.created_at < window_start
    ]
    prev_sessions = [
        session
        for session in repo.list_sessions(patient_id, since=prev_start)
        if session.started_at < window_start
    ]

    current_adherence = adherence(repo, patient_id, since=window_start)
    previous_adherence = adherence(repo, patient_id, since=prev_start)
    prev_settled = previous_adherence["settled"] - current_adherence["settled"]
    prev_taken = previous_adherence["taken"] - current_adherence["taken"]
    previous_pct = (
        round(100 * prev_taken / prev_settled, 1) if prev_settled > 0 else None
    )

    distress_pairs = [
        (session.distress_start, session.distress_end)
        for session in sessions
        if session.distress_start is not None and session.distress_end is not None
    ]
    reliefs = [start - end for start, end in distress_pairs]
    avg_start = _avg([float(s) for s, _ in distress_pairs])
    avg_end = _avg([float(e) for _, e in distress_pairs])

    by_verdict = {verdict.value: 0 for verdict in Verdict}
    for check in checks:
        by_verdict[check.verdict.value] += 1

    by_risk = {level.value: 0 for level in RiskLevel}
    for check in checks:
        by_risk[check.risk_level.value] += 1

    entity_counts: Dict[str, int] = {}
    for check in checks:
        for entity in check.entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
    top_claims = sorted(entity_counts.items(), key=lambda item: item[1], reverse=True)[:5]

    reports = repo.list_reports_for_patient(patient_id, limit=500)
    recent_reports = [report for report in reports if report.created_at >= window_start]
    notes = repo.list_notes(patient_id, limit=50)
    recent_notes = [note for note in notes if note.created_at >= window_start]

    trend = _trend(
        avg_start,
        avg_end,
        _avg([float(s) for s, _ in _distress_pairs(prev_sessions)]),
        _avg([float(e) for _, e in _distress_pairs(prev_sessions)]),
        current_adherence["adherence_pct"],
        previous_pct,
        bool(sessions or checks),
        bool(prev_sessions or prev_checks),
    )

    return {
        "patient_id": patient_id,
        "generated_at": now,
        "window_days": days,
        "window_start": window_start,
        "adherence": current_adherence,
        "previous_adherence_pct": previous_pct,
        "sessions": {
            "count": len(sessions),
            "previous_count": len(prev_sessions),
            "avg_distress_start": avg_start,
            "avg_distress_end": avg_end,
            "avg_relief": _avg([float(relief) for relief in reliefs]),
            "completed": sum(1 for session in sessions if session.status == "closed"),
        },
        "reality_checks": {
            "count": len(checks),
            "previous_count": len(prev_checks),
            "by_verdict": by_verdict,
            "by_risk": by_risk,
            "escalated": sum(1 for check in checks if check.escalated),
            "top_claims": [{"entity": entity, "count": count} for entity, count in top_claims],
        },
        "reports": {
            "count": len(recent_reports),
            "urgent": sum(1 for report in recent_reports if report.urgent),
        },
        "notes": {
            "count": len(recent_notes),
            "latest": recent_notes[0].public() if recent_notes else None,
        },
        "trend": trend,
        "weekly": _weekly_series(repo, patient_id, window_start),
    }


def _distress_pairs(sessions) -> List["tuple[Optional[int], Optional[int]]"]:
    return [
        (session.distress_start, session.distress_end)
        for session in sessions
        if session.distress_start is not None and session.distress_end is not None
    ]


def _trend(
    avg_start: Optional[float],
    avg_end: Optional[float],
    prev_start: Optional[float],
    prev_end: Optional[float],
    adherence_pct: Optional[float],
    prev_adherence_pct: Optional[float],
    has_current: bool,
    has_previous: bool,
) -> Dict[str, Any]:
    """A cautious, explainable direction — never a clinical claim."""
    if not has_current:
        return {"direction": "no_data", "score": None, "explanation": "No activity in this window."}

    score = 0.0
    reasons: List[str] = []

    if avg_start is not None and avg_end is not None:
        if avg_end < avg_start - 0.5:
            score += 1
            reasons.append(f"distress dropped {avg_start:.1f} → {avg_end:.1f} in-session")
        elif avg_end > avg_start + 0.5:
            score -= 1
            reasons.append(f"distress rose {avg_start:.1f} → {avg_end:.1f} in-session")
        else:
            reasons.append("distress flat in-session")

    if prev_end is not None and avg_start is not None:
        if avg_start < prev_end - 0.5:
            score += 1
            reasons.append("starting distress lower than the previous window")
        elif avg_start > prev_end + 0.5:
            score -= 1
            reasons.append("starting distress higher than the previous window")

    if adherence_pct is not None and prev_adherence_pct is not None:
        if adherence_pct >= prev_adherence_pct + 5:
            score += 1
            reasons.append(f"adherence improved {prev_adherence_pct:.0f}% → {adherence_pct:.0f}%")
        elif adherence_pct <= prev_adherence_pct - 5:
            score -= 1
            reasons.append(f"adherence fell {prev_adherence_pct:.0f}% → {adherence_pct:.0f}%")

    if not has_previous:
        direction = "baseline"
        reasons.append("first tracked window; this is the baseline")
    elif score >= 1:
        direction = "improving"
    elif score <= -1:
        direction = "worsening"
    else:
        direction = "stable"

    return {
        "direction": direction,
        "score": round(score, 2),
        "explanation": "; ".join(reasons),
        "disclaimer": "Direction is descriptive of tracked activity, not a clinical assessment.",
    }


def _weekly_series(repo: Repository, patient_id: str, since: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}

    def bucket_for(ts: str) -> Dict[str, Any]:
        key = _bucket_start(ts)
        if key not in buckets:
            buckets[key] = {
                "week_start": key,
                "sessions": 0,
                "reality_checks": 0,
                "high_risk_checks": 0,
                "distress_start": [],
                "distress_end": [],
                "doses_taken": 0,
                "doses_settled": 0,
            }
        return buckets[key]

    for session in repo.list_sessions(patient_id, since=since):
        entry = bucket_for(session.started_at)
        entry["sessions"] += 1
        if session.distress_start is not None:
            entry["distress_start"].append(float(session.distress_start))
        if session.distress_end is not None:
            entry["distress_end"].append(float(session.distress_end))

    for check in repo.list_reality_checks(patient_id, since=since):
        entry = bucket_for(check.created_at)
        entry["reality_checks"] += 1
        if check.risk_level.rank >= RiskLevel.HIGH.rank:
            entry["high_risk_checks"] += 1

    from .medications import adherence as _adherence

    for dose in repo.list_doses(patient_id, due_from=since):
        entry = bucket_for(dose.due_at)
        if dose.status in (DoseStatus.TAKEN, DoseStatus.SKIPPED, DoseStatus.MISSED):
            entry["doses_settled"] += 1
            if dose.status is DoseStatus.TAKEN:
                entry["doses_taken"] += 1

    series: List[Dict[str, Any]] = []
    for key in sorted(buckets):
        entry = buckets[key]
        settled = entry.pop("doses_settled")
        taken = entry.pop("doses_taken")
        entry["adherence_pct"] = round(100 * taken / settled, 1) if settled else None
        entry["avg_distress_start"] = _avg(entry.pop("distress_start"))
        entry["avg_distress_end"] = _avg(entry.pop("distress_end"))
        series.append(entry)
    return series


def care_overview(repo: Repository, user_id: str, role: Role, days: int = 14) -> List[Dict[str, Any]]:
    """One row per patient the caretaker/therapist is assigned to."""
    overview: List[Dict[str, Any]] = []
    for patient in repo.patients_for_user(user_id, role):
        progress = patient_progress(repo, patient.id, days=days)
        latest_note = progress["notes"]["latest"]
        overview.append(
            {
                "patient": patient.public(),
                "trend": progress["trend"],
                "adherence_pct": progress["adherence"]["adherence_pct"],
                "sessions": progress["sessions"]["count"],
                "reality_checks": progress["reality_checks"]["count"],
                "escalations": progress["reality_checks"]["escalated"],
                "avg_relief": progress["sessions"]["avg_relief"],
                "latest_note": latest_note,
            }
        )
    overview.sort(
        key=lambda row: (
            row["escalations"],
            -(row["adherence_pct"] or 0),
        ),
        reverse=True,
    )
    return overview


def patient_home(repo: Repository, patient_id: str) -> Dict[str, Any]:
    """The patient's grounding home screen payload."""
    from .medications import due_reminders

    patient = repo.get_patient(patient_id)
    if not patient:
        raise ValueError("unknown patient")
    progress = patient_progress(repo, patient_id, days=7)
    return {
        "patient": patient.public(),
        "reminders": due_reminders(repo, patient_id),
        "recent_checks": [
            check.public() for check in repo.list_reality_checks(patient_id)[-5:]
        ],
        "adherence_pct": progress["adherence"]["adherence_pct"],
        "trend": progress["trend"],
        "care_team": [member.__dict__ for member in repo.care_team(patient_id)],
        "disclosure": (
            "Shezo describes sensor evidence only. It does not diagnose. If you might act on a "
            "thought, call emergency services."
        ),
    }


def therapist_dashboard(repo: Repository, therapist_id: str, days: int = 14) -> Dict[str, Any]:
    ensure_doses_all(repo)
    patients = care_overview(repo, therapist_id, Role.THERAPIST, days=days)
    return {
        "role": "therapist",
        "generated_at": utcnow(),
        "patients": patients,
        "inbox": inbox(repo, therapist_id),
        "unassigned_alerts": sum(patient["escalations"] for patient in patients),
    }


def caretaker_dashboard(repo: Repository, caretaker_id: str, days: int = 14) -> Dict[str, Any]:
    ensure_doses_all(repo)
    patients = care_overview(repo, caretaker_id, Role.CARETAKER, days=days)
    return {
        "role": "caretaker",
        "generated_at": utcnow(),
        "patients": patients,
        "inbox": inbox(repo, caretaker_id),
        "summary": {
            "patients": len(patients),
            "avg_adherence_pct": _avg(
                [row["adherence_pct"] for row in patients if row["adherence_pct"] is not None]
            ),
            "escalations": sum(row["escalations"] for row in patients),
        },
    }


def ensure_doses_all(repo: Repository) -> None:
    """Materialise today's doses for every patient, so reminders never lag."""
    for patient in _all_patients(repo):
        ensure_doses(repo, patient.id)
        refresh_missed(repo, patient.id)


def _all_patients(repo: Repository) -> List[Any]:
    with repo._lock:  # read-only; keeps the helper out of the public API surface
        rows = repo.conn.execute("SELECT id FROM patients").fetchall()
    return [repo.get_patient(row["id"]) for row in rows]
