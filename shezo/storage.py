"""SQLite persistence for the shared patient record.

Only derived summaries are stored, never raw camera frames or microphone audio.
That is a privacy requirement, not a convenience: the engine receives a frame
sample, extracts numbers/labels, and the sample is discarded.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from typing import Any, Dict, List, Optional, Sequence

from .models import (
    CareTeamMember,
    Dose,
    DoseStatus,
    Medication,
    Note,
    Patient,
    RealityCheck,
    Report,
    ReportKind,
    RiskLevel,
    Role,
    Session,
    User,
    Modality,
    Verdict,
    utcnow,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL,
    password_hash TEXT NOT NULL DEFAULT '',
    salt TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_tokens (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL UNIQUE REFERENCES users(id),
    display_name TEXT NOT NULL,
    therapist_id TEXT REFERENCES users(id),
    primary_caretaker_id TEXT REFERENCES users(id),
    date_of_birth TEXT,
    care_notes TEXT NOT NULL DEFAULT '',
    emergency_contact TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS care_team (
    patient_id TEXT NOT NULL REFERENCES patients(id),
    user_id TEXT NOT NULL REFERENCES users(id),
    role TEXT NOT NULL,
    relationship TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (patient_id, user_id)
);

CREATE TABLE IF NOT EXISTS medications (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    name TEXT NOT NULL,
    dose TEXT NOT NULL,
    times_json TEXT NOT NULL,
    instructions TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS doses (
    id TEXT PRIMARY KEY,
    medication_id TEXT NOT NULL REFERENCES medications(id),
    patient_id TEXT NOT NULL REFERENCES patients(id),
    due_at TEXT NOT NULL,
    status TEXT NOT NULL,
    acted_at TEXT,
    UNIQUE (medication_id, due_at)
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    distress_start INTEGER,
    distress_end INTEGER,
    camera_consent INTEGER NOT NULL DEFAULT 0,
    mic_consent INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS reality_checks (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    session_id TEXT,
    created_at TEXT NOT NULL,
    claim TEXT NOT NULL,
    modality TEXT NOT NULL,
    verdict TEXT NOT NULL,
    confidence REAL NOT NULL,
    risk_level TEXT NOT NULL,
    headline TEXT NOT NULL,
    explanation TEXT NOT NULL,
    sensor_summary TEXT NOT NULL,
    grounding_steps_json TEXT NOT NULL,
    entities_json TEXT NOT NULL,
    escalated INTEGER NOT NULL DEFAULT 0,
    disclosure TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    author_role TEXT NOT NULL,
    author_id TEXT NOT NULL,
    session_id TEXT,
    reality_check_id TEXT,
    payload_json TEXT NOT NULL,
    urgent INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS report_recipients (
    report_id TEXT NOT NULL REFERENCES reports(id),
    user_id TEXT NOT NULL REFERENCES users(id),
    role TEXT NOT NULL,
    delivered_at TEXT,
    read_at TEXT,
    acknowledged_at TEXT,
    PRIMARY KEY (report_id, user_id)
);

CREATE TABLE IF NOT EXISTS deliveries (
    id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL REFERENCES reports(id),
    user_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    author_id TEXT NOT NULL,
    author_role TEXT NOT NULL,
    body TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit (
    id TEXT PRIMARY KEY,
    actor_id TEXT,
    action TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    detail_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_doses_patient_due ON doses (patient_id, due_at);
CREATE INDEX IF NOT EXISTS idx_checks_patient ON reality_checks (patient_id, created_at);
CREATE INDEX IF NOT EXISTS idx_reports_patient ON reports (patient_id, created_at);
CREATE INDEX IF NOT EXISTS idx_notes_patient ON notes (patient_id, created_at);
"""


def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex
    return f"{prefix}_{raw}" if prefix else raw


def _js(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _jl(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


class Repository:
    """Thread-safe-enough SQLite repository (serialised with a lock)."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ---------------------------------------------------------------- users
    def add_user(self, user: User) -> User:
        with self._lock:
            self.conn.execute(
                "INSERT INTO users (id,email,display_name,role,password_hash,salt,created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    user.id,
                    user.email.lower().strip(),
                    user.display_name,
                    user.role.value,
                    user.password_hash,
                    user.salt,
                    user.created_at,
                ),
            )
            self.conn.commit()
        return user

    def _user(self, row: sqlite3.Row) -> User:
        return User(
            id=row["id"],
            email=row["email"],
            display_name=row["display_name"],
            role=Role(row["role"]),
            created_at=row["created_at"],
            password_hash=row["password_hash"],
            salt=row["salt"],
        )

    def get_user(self, user_id: str) -> Optional[User]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return self._user(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[User]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
            ).fetchone()
        return self._user(row) if row else None

    # --------------------------------------------------------------- tokens
    def save_token(self, token: str, user_id: str, expires_at: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO auth_tokens (token,user_id,created_at,expires_at) VALUES (?,?,?,?)",
                (token, user_id, utcnow(), expires_at),
            )
            self.conn.commit()

    def token_user(self, token: str) -> Optional[User]:
        with self._lock:
            row = self.conn.execute(
                "SELECT u.* FROM auth_tokens t JOIN users u ON u.id = t.user_id WHERE t.token = ?",
                (token,),
            ).fetchone()
        return self._user(row) if row else None

    def delete_token(self, token: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
            self.conn.commit()

    # ------------------------------------------------------------- patients
    def add_patient(self, patient: Patient) -> Patient:
        with self._lock:
            self.conn.execute(
                "INSERT INTO patients (id,user_id,display_name,therapist_id,primary_caretaker_id,"
                "date_of_birth,care_notes,emergency_contact,created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    patient.id,
                    patient.user_id,
                    patient.display_name,
                    patient.therapist_id,
                    patient.primary_caretaker_id,
                    patient.date_of_birth,
                    patient.care_notes,
                    patient.emergency_contact,
                    patient.created_at,
                ),
            )
            self.conn.commit()
        return patient

    def update_patient(self, patient: Patient) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE patients SET therapist_id=?, primary_caretaker_id=?, care_notes=?,"
                " emergency_contact=?, display_name=? WHERE id=?",
                (
                    patient.therapist_id,
                    patient.primary_caretaker_id,
                    patient.care_notes,
                    patient.emergency_contact,
                    patient.display_name,
                    patient.id,
                ),
            )
            self.conn.commit()

    def _patient(self, row: sqlite3.Row) -> Patient:
        return Patient(
            id=row["id"],
            user_id=row["user_id"],
            display_name=row["display_name"],
            created_at=row["created_at"],
            therapist_id=row["therapist_id"],
            primary_caretaker_id=row["primary_caretaker_id"],
            date_of_birth=row["date_of_birth"],
            care_notes=row["care_notes"],
            emergency_contact=row["emergency_contact"],
        )

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
        return self._patient(row) if row else None

    def patient_for_user(self, user_id: str) -> Optional[Patient]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM patients WHERE user_id = ?", (user_id,)).fetchone()
        return self._patient(row) if row else None

    def add_care_member(self, member: CareTeamMember) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO care_team (patient_id,user_id,role,relationship)"
                " VALUES (?,?,?,?)",
                (member.patient_id, member.user_id, member.role.value, member.relationship),
            )
            self.conn.commit()

    def care_team(self, patient_id: str) -> List[CareTeamMember]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT c.*, u.display_name FROM care_team c"
                " LEFT JOIN users u ON u.id = c.user_id WHERE c.patient_id = ?",
                (patient_id,),
            ).fetchall()
        return [
            CareTeamMember(
                patient_id=row["patient_id"],
                user_id=row["user_id"],
                role=Role(row["role"]),
                relationship=row["relationship"],
                display_name=row["display_name"] or "",
            )
            for row in rows
        ]

    def patients_for_user(self, user_id: str, role: Role) -> List[Patient]:
        """Patients a caretaker/therapist may see, plus their own record."""
        with self._lock:
            if role is Role.PATIENT:
                rows = self.conn.execute(
                    "SELECT * FROM patients WHERE user_id = ?", (user_id,)
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT p.* FROM patients p"
                    " JOIN care_team c ON c.patient_id = p.id WHERE c.user_id = ?"
                    " ORDER BY p.display_name",
                    (user_id,),
                ).fetchall()
        return [self._patient(row) for row in rows]

    def has_access(self, patient_id: str, user_id: str, role: Role) -> bool:
        if role is Role.PATIENT:
            patient = self.get_patient(patient_id)
            return bool(patient and patient.user_id == user_id)
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM care_team WHERE patient_id = ? AND user_id = ?",
                (patient_id, user_id),
            ).fetchone()
        return row is not None

    # ---------------------------------------------------------- medications
    def add_medication(self, med: Medication) -> Medication:
        with self._lock:
            self.conn.execute(
                "INSERT INTO medications (id,patient_id,name,dose,times_json,instructions,active,created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    med.id,
                    med.patient_id,
                    med.name,
                    med.dose,
                    _js(med.times),
                    med.instructions,
                    1 if med.active else 0,
                    med.created_at or utcnow(),
                ),
            )
            self.conn.commit()
        return med

    def _medication(self, row: sqlite3.Row) -> Medication:
        return Medication(
            id=row["id"],
            patient_id=row["patient_id"],
            name=row["name"],
            dose=row["dose"],
            times=list(_jl(row["times_json"]) or []),
            instructions=row["instructions"],
            active=bool(row["active"]),
            created_at=row["created_at"],
        )

    def list_medications(self, patient_id: str, active_only: bool = True) -> List[Medication]:
        query = "SELECT * FROM medications WHERE patient_id = ?"
        if active_only:
            query += " AND active = 1"
        with self._lock:
            rows = self.conn.execute(query + " ORDER BY name", (patient_id,)).fetchall()
        return [self._medication(row) for row in rows]

    def get_medication(self, medication_id: str) -> Optional[Medication]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM medications WHERE id = ?", (medication_id,)).fetchone()
        return self._medication(row) if row else None

    # --------------------------------------------------------------- doses
    def upsert_dose(self, dose: Dose) -> Dose:
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO doses (id,medication_id,patient_id,due_at,status,acted_at)"
                " VALUES (?,?,?,?,?,?)",
                (
                    dose.id,
                    dose.medication_id,
                    dose.patient_id,
                    dose.due_at,
                    dose.status.value,
                    dose.acted_at,
                ),
            )
            self.conn.commit()
        return dose

    def _dose(self, row: sqlite3.Row) -> Dose:
        return Dose(
            id=row["id"],
            medication_id=row["medication_id"],
            patient_id=row["patient_id"],
            due_at=row["due_at"],
            status=DoseStatus(row["status"]),
            acted_at=row["acted_at"],
            medication_name=row["name"] if "name" in row.keys() else "",
            medication_dose=row["dose_text"] if "dose_text" in row.keys() else "",
        )

    _DOSE_SELECT = (
        "SELECT d.*, m.name AS name, m.dose AS dose_text FROM doses d"
        " JOIN medications m ON m.id = d.medication_id"
    )

    def get_dose(self, dose_id: str) -> Optional[Dose]:
        with self._lock:
            row = self.conn.execute(self._DOSE_SELECT + " WHERE d.id = ?", (dose_id,)).fetchone()
        return self._dose(row) if row else None

    def list_doses(
        self,
        patient_id: str,
        due_from: Optional[str] = None,
        due_to: Optional[str] = None,
        statuses: Optional[Sequence[DoseStatus]] = None,
    ) -> List[Dose]:
        query = self._DOSE_SELECT + " WHERE d.patient_id = ?"
        params: List[Any] = [patient_id]
        if due_from:
            query += " AND d.due_at >= ?"
            params.append(due_from)
        if due_to:
            query += " AND d.due_at <= ?"
            params.append(due_to)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND d.status IN ({placeholders})"
            params.extend(status.value for status in statuses)
        query += " ORDER BY d.due_at"
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [self._dose(row) for row in rows]

    def set_dose_status(self, dose_id: str, status: DoseStatus, acted_at: Optional[str]) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE doses SET status = ?, acted_at = ? WHERE id = ?",
                (status.value, acted_at, dose_id),
            )
            self.conn.commit()

    # ------------------------------------------------------------ sessions
    def add_session(self, session: Session) -> Session:
        with self._lock:
            self.conn.execute(
                "INSERT INTO sessions (id,patient_id,started_at,ended_at,distress_start,distress_end,"
                "camera_consent,mic_consent,status) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    session.id,
                    session.patient_id,
                    session.started_at,
                    session.ended_at,
                    session.distress_start,
                    session.distress_end,
                    1 if session.camera_consent else 0,
                    1 if session.mic_consent else 0,
                    session.status,
                ),
            )
            self.conn.commit()
        return session

    def _session(self, row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            patient_id=row["patient_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            distress_start=row["distress_start"],
            distress_end=row["distress_end"],
            camera_consent=bool(row["camera_consent"]),
            mic_consent=bool(row["mic_consent"]),
            status=row["status"],
        )

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._session(row) if row else None

    def update_session(self, session: Session) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE sessions SET ended_at=?, distress_end=?, status=? WHERE id=?",
                (session.ended_at, session.distress_end, session.status, session.id),
            )
            self.conn.commit()

    def list_sessions(self, patient_id: str, since: Optional[str] = None) -> List[Session]:
        query = "SELECT * FROM sessions WHERE patient_id = ?"
        params: List[Any] = [patient_id]
        if since:
            query += " AND started_at >= ?"
            params.append(since)
        query += " ORDER BY started_at"
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [self._session(row) for row in rows]

    # ------------------------------------------------------- reality checks
    def add_reality_check(self, check: RealityCheck) -> RealityCheck:
        with self._lock:
            self.conn.execute(
                "INSERT INTO reality_checks (id,patient_id,session_id,created_at,claim,modality,verdict,"
                "confidence,risk_level,headline,explanation,sensor_summary,grounding_steps_json,"
                "entities_json,escalated,disclosure) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    check.id,
                    check.patient_id,
                    check.session_id,
                    check.created_at,
                    check.claim,
                    check.modality.value,
                    check.verdict.value,
                    check.confidence,
                    check.risk_level.value,
                    check.headline,
                    check.explanation,
                    check.sensor_summary,
                    _js(check.grounding_steps),
                    _js(check.entities),
                    1 if check.escalated else 0,
                    check.disclosure,
                ),
            )
            self.conn.commit()
        return check

    def _check(self, row: sqlite3.Row) -> RealityCheck:
        return RealityCheck(
            id=row["id"],
            patient_id=row["patient_id"],
            session_id=row["session_id"],
            created_at=row["created_at"],
            claim=row["claim"],
            modality=Modality(row["modality"]),
            verdict=Verdict(row["verdict"]),
            confidence=row["confidence"],
            risk_level=RiskLevel(row["risk_level"]),
            headline=row["headline"],
            explanation=row["explanation"],
            sensor_summary=row["sensor_summary"],
            grounding_steps=list(_jl(row["grounding_steps_json"]) or []),
            entities=list(_jl(row["entities_json"]) or []),
            escalated=bool(row["escalated"]),
            disclosure=row["disclosure"],
        )

    def get_reality_check(self, check_id: str) -> Optional[RealityCheck]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM reality_checks WHERE id = ?", (check_id,)).fetchone()
        return self._check(row) if row else None

    def list_reality_checks(self, patient_id: str, since: Optional[str] = None) -> List[RealityCheck]:
        query = "SELECT * FROM reality_checks WHERE patient_id = ?"
        params: List[Any] = [patient_id]
        if since:
            query += " AND created_at >= ?"
            params.append(since)
        query += " ORDER BY created_at"
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [self._check(row) for row in rows]

    # ------------------------------------------------------------- reports
    def add_report(self, report: Report) -> Report:
        with self._lock:
            self.conn.execute(
                "INSERT INTO reports (id,patient_id,kind,title,summary,author_role,author_id,"
                "session_id,reality_check_id,payload_json,urgent,created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    report.id,
                    report.patient_id,
                    report.kind.value,
                    report.title,
                    report.summary,
                    report.author_role.value,
                    report.author_id,
                    report.session_id,
                    report.reality_check_id,
                    _js(report.payload),
                    1 if report.urgent else 0,
                    report.created_at,
                ),
            )
            self.conn.commit()
        return report

    def add_recipient(self, report_id: str, user_id: str, role: Role) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO report_recipients (report_id,user_id,role) VALUES (?,?,?)",
                (report_id, user_id, role.value),
            )
            self.conn.commit()

    def mark_delivered(self, report_id: str, user_id: str, at: Optional[str] = None) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE report_recipients SET delivered_at = ? WHERE report_id = ? AND user_id = ?",
                (at or utcnow(), report_id, user_id),
            )
            self.conn.commit()

    def mark_read(self, report_id: str, user_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE report_recipients SET read_at = COALESCE(read_at, ?)"
                " WHERE report_id = ? AND user_id = ?",
                (utcnow(), report_id, user_id),
            )
            self.conn.commit()

    def acknowledge(self, report_id: str, user_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute(
                "UPDATE report_recipients SET acknowledged_at = ?, read_at = COALESCE(read_at, ?)"
                " WHERE report_id = ? AND user_id = ?",
                (utcnow(), utcnow(), report_id, user_id),
            )
            self.conn.commit()
        return cur.rowcount > 0

    def _report(self, row: sqlite3.Row) -> Report:
        return Report(
            id=row["id"],
            patient_id=row["patient_id"],
            kind=ReportKind(row["kind"]),
            title=row["title"],
            summary=row["summary"],
            author_role=Role(row["author_role"]),
            author_id=row["author_id"],
            created_at=row["created_at"],
            payload=dict(_jl(row["payload_json"]) or {}),
            session_id=row["session_id"],
            reality_check_id=row["reality_check_id"],
            urgent=bool(row["urgent"]),
        )

    def _recipients_for(self, report_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT user_id, role, delivered_at, read_at, acknowledged_at"
                " FROM report_recipients WHERE report_id = ?",
                (report_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_reports_for_user(
        self, user_id: str, unread_only: bool = False, limit: int = 100
    ) -> List[Report]:
        query = (
            "SELECT r.*, rr.read_at AS read_at, rr.acknowledged_at AS acknowledged_at"
            " FROM reports r JOIN report_recipients rr ON rr.report_id = r.id"
            " WHERE rr.user_id = ?"
        )
        params: List[Any] = [user_id]
        if unread_only:
            query += " AND rr.read_at IS NULL"
        query += " ORDER BY r.created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        reports: List[Report] = []
        for row in rows:
            report = self._report(row)
            report.recipients = self._recipients_for(report.id)
            reports.append(report)
        return reports

    def list_reports_for_patient(self, patient_id: str, limit: int = 200) -> List[Report]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM reports WHERE patient_id = ? ORDER BY created_at DESC LIMIT ?",
                (patient_id, limit),
            ).fetchall()
        reports = [self._report(row) for row in rows]
        for report in reports:
            report.recipients = self._recipients_for(report.id)
        return reports

    def get_report(self, report_id: str) -> Optional[Report]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if not row:
            return None
        report = self._report(row)
        report.recipients = self._recipients_for(report_id)
        return report

    def report_recipient_ids(self, report_id: str) -> List[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT user_id FROM report_recipients WHERE report_id = ?", (report_id,)
            ).fetchall()
        return [row["user_id"] for row in rows]

    # ------------------------------------------------------- notifications
    def add_delivery(
        self, report_id: str, user_id: str, channel: str, status: str, detail: str = ""
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO deliveries (id,report_id,user_id,channel,status,detail,created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (new_id("dlv"), report_id, user_id, channel, status, detail, utcnow()),
            )
            self.conn.commit()

    def deliveries_for(self, report_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM deliveries WHERE report_id = ? ORDER BY created_at", (report_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    # --------------------------------------------------------------- notes
    def add_note(self, note: Note) -> Note:
        with self._lock:
            self.conn.execute(
                "INSERT INTO notes (id,patient_id,author_id,author_role,body,pinned,created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    note.id,
                    note.patient_id,
                    note.author_id,
                    note.author_role.value,
                    note.body,
                    1 if note.pinned else 0,
                    note.created_at,
                ),
            )
            self.conn.commit()
        return note

    def list_notes(self, patient_id: str, limit: int = 100) -> List[Note]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT n.*, u.display_name AS author_name FROM notes n"
                " LEFT JOIN users u ON u.id = n.author_id"
                " WHERE n.patient_id = ? ORDER BY n.pinned DESC, n.created_at DESC LIMIT ?",
                (patient_id, limit),
            ).fetchall()
        return [
            Note(
                id=row["id"],
                patient_id=row["patient_id"],
                author_id=row["author_id"],
                author_role=Role(row["author_role"]),
                author_name=row["author_name"] or "",
                body=row["body"],
                created_at=row["created_at"],
                pinned=bool(row["pinned"]),
            )
            for row in rows
        ]

    # --------------------------------------------------------------- audit
    def audit(self, actor_id: Optional[str], action: str, subject: str = "", detail: Any = None) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO audit (id,actor_id,action,subject,detail_json,created_at) VALUES (?,?,?,?,?,?)",
                (new_id("aud"), actor_id, action, subject, _js(detail or {}), utcnow()),
            )
            self.conn.commit()

    def audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM audit ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "action": row["action"],
                "actor_id": row["actor_id"],
                "subject": row["subject"],
                "detail": _jl(row["detail_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
