"""Optional MySQL / MariaDB backend (the database behind phpMyAdmin).

The repository code is written once, against the tiny slice of the ``sqlite3``
connection API it actually uses: ``execute(sql, params)`` returning a cursor with
``fetchone`` / ``fetchall`` / ``rowcount``, rows addressable by column name, plus
``executescript`` and ``commit``. :class:`MysqlConnection` provides exactly that
on top of PyMySQL, translating the handful of SQLite spellings the schema uses.

That keeps one repository and one set of queries, so the model cannot behave
differently depending on where the shared patient record lives.

Configuration is a DSN, e.g. ``mysql://root@127.0.0.1:3306/shezo``.
PyMySQL is an optional dependency: it is only imported when MySQL is used.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

DEFAULT_DSN = "mysql://root@127.0.0.1:3306/shezo"
DEFAULT_CHARSET = "utf8mb4"

#: MariaDB/MySQL schema mirroring ``storage.SCHEMA``. Indexes are declared inline
#: because MariaDB has no ``CREATE INDEX IF NOT EXISTS``.
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(64) NOT NULL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    display_name VARCHAR(255) NOT NULL,
    role VARCHAR(32) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    salt VARCHAR(64) NOT NULL,
    created_at VARCHAR(40) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS auth_tokens (
    token VARCHAR(128) NOT NULL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    expires_at VARCHAR(40) NOT NULL,
    KEY idx_tokens_user (user_id),
    CONSTRAINT fk_tokens_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS patients (
    id VARCHAR(64) NOT NULL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL UNIQUE,
    display_name VARCHAR(255) NOT NULL,
    therapist_id VARCHAR(64) NULL,
    primary_caretaker_id VARCHAR(64) NULL,
    date_of_birth VARCHAR(40) NULL,
    care_notes LONGTEXT NOT NULL,
    emergency_contact LONGTEXT NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    CONSTRAINT fk_patients_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS care_team (
    patient_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64) NOT NULL,
    role VARCHAR(32) NOT NULL,
    relationship VARCHAR(255) NOT NULL,
    PRIMARY KEY (patient_id, user_id),
    KEY idx_care_team_user (user_id),
    CONSTRAINT fk_care_patient FOREIGN KEY (patient_id) REFERENCES patients(id),
    CONSTRAINT fk_care_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS medications (
    id VARCHAR(64) NOT NULL PRIMARY KEY,
    patient_id VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    dose VARCHAR(255) NOT NULL,
    times_json LONGTEXT NOT NULL,
    instructions LONGTEXT NOT NULL,
    active TINYINT NOT NULL DEFAULT 1,
    created_at VARCHAR(40) NOT NULL,
    KEY idx_meds_patient (patient_id),
    CONSTRAINT fk_meds_patient FOREIGN KEY (patient_id) REFERENCES patients(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS doses (
    id VARCHAR(64) NOT NULL PRIMARY KEY,
    medication_id VARCHAR(64) NOT NULL,
    patient_id VARCHAR(64) NOT NULL,
    due_at VARCHAR(40) NOT NULL,
    status VARCHAR(32) NOT NULL,
    acted_at VARCHAR(40) NULL,
    UNIQUE KEY uq_dose_slot (medication_id, due_at),
    KEY idx_doses_patient_due (patient_id, due_at),
    CONSTRAINT fk_doses_med FOREIGN KEY (medication_id) REFERENCES medications(id),
    CONSTRAINT fk_doses_patient FOREIGN KEY (patient_id) REFERENCES patients(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS sessions (
    id VARCHAR(64) NOT NULL PRIMARY KEY,
    patient_id VARCHAR(64) NOT NULL,
    started_at VARCHAR(40) NOT NULL,
    ended_at VARCHAR(40) NULL,
    distress_start INT NULL,
    distress_end INT NULL,
    camera_consent TINYINT NOT NULL DEFAULT 0,
    mic_consent TINYINT NOT NULL DEFAULT 0,
    status VARCHAR(32) NOT NULL DEFAULT 'open',
    KEY idx_sessions_patient (patient_id, started_at),
    CONSTRAINT fk_sessions_patient FOREIGN KEY (patient_id) REFERENCES patients(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS reality_checks (
    id VARCHAR(64) NOT NULL PRIMARY KEY,
    patient_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64) NULL,
    created_at VARCHAR(40) NOT NULL,
    claim LONGTEXT NOT NULL,
    modality VARCHAR(32) NOT NULL,
    verdict VARCHAR(32) NOT NULL,
    confidence DOUBLE NOT NULL,
    risk_level VARCHAR(32) NOT NULL,
    headline LONGTEXT NOT NULL,
    explanation LONGTEXT NOT NULL,
    sensor_summary LONGTEXT NOT NULL,
    grounding_steps_json LONGTEXT NOT NULL,
    entities_json LONGTEXT NOT NULL,
    escalated TINYINT NOT NULL DEFAULT 0,
    disclosure LONGTEXT NOT NULL,
    KEY idx_checks_patient (patient_id, created_at),
    CONSTRAINT fk_checks_patient FOREIGN KEY (patient_id) REFERENCES patients(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS reports (
    id VARCHAR(64) NOT NULL PRIMARY KEY,
    patient_id VARCHAR(64) NOT NULL,
    kind VARCHAR(32) NOT NULL,
    title VARCHAR(255) NOT NULL,
    summary LONGTEXT NOT NULL,
    author_role VARCHAR(32) NOT NULL,
    author_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64) NULL,
    reality_check_id VARCHAR(64) NULL,
    payload_json LONGTEXT NOT NULL,
    urgent TINYINT NOT NULL DEFAULT 0,
    created_at VARCHAR(40) NOT NULL,
    KEY idx_reports_patient (patient_id, created_at),
    CONSTRAINT fk_reports_patient FOREIGN KEY (patient_id) REFERENCES patients(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS report_recipients (
    report_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64) NOT NULL,
    role VARCHAR(32) NOT NULL,
    delivered_at VARCHAR(40) NULL,
    read_at VARCHAR(40) NULL,
    acknowledged_at VARCHAR(40) NULL,
    PRIMARY KEY (report_id, user_id),
    KEY idx_recipients_user (user_id),
    CONSTRAINT fk_recipients_report FOREIGN KEY (report_id) REFERENCES reports(id),
    CONSTRAINT fk_recipients_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS deliveries (
    id VARCHAR(64) NOT NULL PRIMARY KEY,
    report_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(64) NOT NULL,
    channel VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL,
    detail LONGTEXT NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    KEY idx_deliveries_report (report_id),
    CONSTRAINT fk_deliveries_report FOREIGN KEY (report_id) REFERENCES reports(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS notes (
    id VARCHAR(64) NOT NULL PRIMARY KEY,
    patient_id VARCHAR(64) NOT NULL,
    author_id VARCHAR(64) NOT NULL,
    author_role VARCHAR(32) NOT NULL,
    body LONGTEXT NOT NULL,
    pinned TINYINT NOT NULL DEFAULT 0,
    created_at VARCHAR(40) NOT NULL,
    KEY idx_notes_patient (patient_id, created_at),
    CONSTRAINT fk_notes_patient FOREIGN KEY (patient_id) REFERENCES patients(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS audit (
    id VARCHAR(64) NOT NULL PRIMARY KEY,
    actor_id VARCHAR(64) NULL,
    action VARCHAR(128) NOT NULL,
    subject VARCHAR(128) NOT NULL,
    detail_json LONGTEXT NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    KEY idx_audit_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_OR_REPLACE = re.compile(r"INSERT\s+OR\s+REPLACE", re.IGNORECASE)
_OR_IGNORE = re.compile(r"INSERT\s+OR\s+IGNORE", re.IGNORECASE)


class MysqlUnavailable(RuntimeError):
    """Raised when MySQL is requested but cannot be reached or used."""


def parse_dsn(dsn: str) -> Dict[str, Any]:
    """Parse ``mysql://user:password@host:port/database`` into connect args."""
    parsed = urlparse(dsn.strip())
    if parsed.scheme not in ("mysql", "mariadb", "mysql+pymysql"):
        raise ValueError(f"unsupported database URL scheme: {parsed.scheme!r} (use mysql://)")
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 3306)
    user = unquote(parsed.username or "root")
    password = unquote(parsed.password or "")
    database = (parsed.path or "/shezo").lstrip("/") or "shezo"
    if not re.fullmatch(r"[A-Za-z0-9_]+", database):
        raise ValueError(f"invalid database name in URL: {database!r}")
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
    }


def _translate(sql: str) -> str:
    """Rewrite the SQLite dialect used by the repository into MySQL."""
    sql = _OR_REPLACE.sub("REPLACE", sql)
    sql = _OR_IGNORE.sub("INSERT IGNORE", sql)
    return sql.replace("?", "%s")


def _split_statements(script: str) -> List[str]:
    return [chunk.strip() for chunk in script.split(";") if chunk.strip()]


class _Cursor:
    """Thin wrapper exposing the attributes the repository relies on."""

    __slots__ = ("_cur",)

    def __init__(self, cur: Any) -> None:
        self._cur = cur

    def fetchone(self) -> Optional[Dict[str, Any]]:
        return self._cur.fetchone()

    def fetchall(self) -> List[Dict[str, Any]]:
        return list(self._cur.fetchall())

    @property
    def rowcount(self) -> int:
        return int(self._cur.rowcount)

    def close(self) -> None:
        self._cur.close()


class MysqlConnection:
    """A ``sqlite3.Connection``-shaped handle over one MySQL connection."""

    def __init__(self, dsn: str = DEFAULT_DSN, connect_timeout: int = 6) -> None:
        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise MysqlUnavailable(
                "MySQL storage needs PyMySQL: py -m pip install pymysql"
            ) from exc

        self._pymysql = pymysql
        self._dict_cursor = DictCursor
        self._lock = threading.RLock()
        self._last_use = 0.0
        self.config = parse_dsn(dsn)
        self.dsn = dsn

        # Create the schema-up database on first run (phpMyAdmin-friendly).
        try:
            admin = pymysql.connect(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["user"],
                password=self.config["password"],
                charset=DEFAULT_CHARSET,
                autocommit=True,
                connect_timeout=connect_timeout,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced with a helpful message
            raise MysqlUnavailable(
                f"cannot reach MySQL at {self.config['host']}:{self.config['port']}: {exc}"
            ) from exc
        try:
            with admin.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.config['database']}`"
                    " CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
        finally:
            admin.close()

        try:
            self._conn = pymysql.connect(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["user"],
                password=self.config["password"],
                database=self.config["database"],
                charset=DEFAULT_CHARSET,
                cursorclass=self._dict_cursor,
                autocommit=True,
                connect_timeout=connect_timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise MysqlUnavailable(f"cannot open database: {exc}") from exc

    @property
    def database(self) -> str:
        return self.config["database"]

    def _connection(self):
        # A local connection can be dropped between requests; reconnect lazily.
        if time.monotonic() - self._last_use > 45:
            try:
                self._conn.ping(reconnect=True)
            except Exception:  # noqa: BLE001
                pass
        self._last_use = time.monotonic()
        return self._conn

    def execute(self, sql: str, params: Sequence[Any] = ()) -> _Cursor:
        with self._lock:
            conn = self._connection()
            cur = conn.cursor()
            try:
                cur.execute(_translate(sql), tuple(params) or None)
            except Exception:
                cur.close()
                raise
        return _Cursor(cur)

    def executescript(self, script: str) -> None:
        with self._lock:
            conn = self._connection()
            with conn.cursor() as cur:
                for statement in _split_statements(script):
                    if statement.upper().startswith("PRAGMA"):
                        continue
                    cur.execute(statement)

    def commit(self) -> None:
        """Autocommit is on; kept so the repository code is backend-agnostic."""

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass

    def cursor(self):
        return self._connection().cursor()

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - convenience
        return getattr(self._connection(), name)


def connect(dsn: str = DEFAULT_DSN) -> MysqlConnection:
    conn = MysqlConnection(dsn)
    conn.executescript(SCHEMA)
    return conn
