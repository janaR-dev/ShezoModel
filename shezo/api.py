"""JSON API for the three interfaces (no HTML, no UI).

Routes are grouped by role. Role checks are enforced here, on the server, and
patient access is re-verified against the care team for every patient-scoped
path — a caretaker cannot read another family's record by guessing an id.

Run locally:  py -m shezo.api        (SHEZO_DB, SHEZO_PORT, SHEZO_HOST)
"""

from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .ai.engine import GroundingEngine
from .ai.perception import audio_from_dict, frame_from_dict
from .auth import AuthError, AuthService, RoleError
from .models import Medication, Note, Role, User, utcnow
from .services import medications, progress, reports as report_service
from .services.notifications import build_gateway
from .storage import Repository, new_id, open_backend

MAX_BODY_BYTES = 256 * 1024
ALL_ROLES = (Role.PATIENT, Role.CARETAKER, Role.THERAPIST)

#: Static front-end (plain HTML/CSS/JS) served by this same process, so the
#: browser talks to the API on its own origin and no CORS setup is needed.
WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".map": "application/json; charset=utf-8",
}


@dataclass
class AppContext:
    repo: Repository
    auth: AuthService
    engine: GroundingEngine
    gateway: Any = None


def build_context(db_path: str = ":memory:", llm: Any = None) -> AppContext:
    repo = Repository(db_path)
    gateway = build_gateway()
    return AppContext(
        repo=repo,
        auth=AuthService(repo),
        engine=GroundingEngine(repo, llm=llm, gateway=gateway),
        gateway=gateway,
    )


# --------------------------------------------------------------------- routing
Route = Tuple[str, "re.Pattern[str]", Callable[..., Dict[str, Any]], Tuple[Role, ...]]
ROUTES: List[Route] = []


def route(method: str, pattern: str, roles: Tuple[Role, ...] = ALL_ROLES):
    compiled = re.compile(pattern)

    def decorator(fn):
        ROUTES.append((method.upper(), compiled, fn, roles))
        return fn

    return decorator


# -------------------------------------------------------------------- helpers
def _patient(ctx: AppContext, user: User):
    patient = ctx.repo.patient_for_user(user.id)
    if not patient:
        raise PermissionError("this account has no patient record")
    return patient


def _guard(ctx: AppContext, user: User, patient_id: str):
    if not ctx.repo.has_access(patient_id, user.id, user.role):
        raise PermissionError("you are not assigned to this patient")
    patient = ctx.repo.get_patient(patient_id)
    if not patient:
        raise ValueError("unknown patient")
    return patient


def _session_for(ctx: AppContext, user: User, patient_id: str, session_id: str):
    session = ctx.repo.get_session(session_id)
    if not session or session.patient_id != patient_id:
        raise ValueError("unknown session")
    if session.status != "open":
        raise ValueError("this session is already closed")
    return session


# ----------------------------------------------------------------------- auth
@route("GET", r"/api/health", ())
def health(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    return {"status": "ok", "time": utcnow(), "ai": "ready"}


@route("POST", r"/api/auth/login", ())
def login(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    token_value, account = ctx.auth.login(body.get("email", ""), body.get("password", ""))
    return {"token": token_value, "user": account.public()}


@route("POST", r"/api/auth/logout")
def logout(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    ctx.auth.logout(token)
    return {"ok": True}


@route("GET", r"/api/auth/me")
def me(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    return {"user": user.public()}


# -------------------------------------------------------------------- patient
@route("GET", r"/api/patient/home", (Role.PATIENT,))
def patient_home(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _patient(ctx, user)
    payload = progress.patient_home(ctx.repo, patient.id)
    # The browser cannot remember a session across a reload, so the open one is
    # handed back here. Idle sessions are closed (with their summary) on the way.
    open_session = ctx.engine.resume_session(patient.id)
    payload["open_session"] = open_session.public() if open_session else None
    return payload


@route("GET", r"/api/patient/progress", (Role.PATIENT,))
def patient_progress_view(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _patient(ctx, user)
    return progress.patient_progress(ctx.repo, patient.id)


@route("GET", r"/api/patient/reports", (Role.PATIENT,))
def patient_reports(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _patient(ctx, user)
    items = []
    for report in ctx.repo.list_reports_for_patient(patient.id):
        if report.kind.value in ("reality_check", "session_summary"):
            items.append(report.public())
    return {"reports": items[:100], "note": f"{len(items)} report(s) shared with your care team"}


@route("POST", r"/api/patient/sessions", (Role.PATIENT,))
def start_session(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _patient(ctx, user)
    distress = body.get("distress_start")
    camera_consent = bool(body.get("camera_consent"))
    mic_consent = bool(body.get("mic_consent"))

    # A reload (or a second tab) must not fork the record into two sessions.
    existing = ctx.engine.resume_session(patient.id)
    if existing:
        session = ctx.engine.adopt_session(
            existing,
            distress=int(distress) if distress is not None else None,
            camera_consent=camera_consent,
            mic_consent=mic_consent,
        )
        ctx.repo.audit(user.id, "session.resumed", session.id, session.public())
        return {
            "session": session.public(),
            "resumed": True,
            "notice": "Your open grounding session was picked up again, so this check continues on the same record.",
        }

    session = ctx.engine.open_session(
        patient.id,
        distress=int(distress) if distress is not None else None,
        camera_consent=camera_consent,
        mic_consent=mic_consent,
    )
    ctx.repo.audit(user.id, "session.opened", session.id, session.public())
    return {
        "session": session.public(),
        "resumed": False,
        "notice": "Camera and microphone are read only while this session is open, and only with the consent you gave.",
    }


@route("POST", r"/api/patient/sessions/([^/]+)/checks", (Role.PATIENT,))
def create_check(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _patient(ctx, user)
    session = _session_for(ctx, user, patient.id, params[0])
    if ctx.engine.session_expired(session):
        # Never write checks onto a session nobody is sitting with; close it so
        # the care team still gets a summary, then ask for a fresh session.
        ctx.engine.expire_session(session)
        raise ValueError(
            "this session timed out after inactivity and was closed. Start a new one."
        )
    distress = body.get("distress")

    outcome = ctx.engine.run_reality_check(
        patient_id=patient.id,
        claim_text=str(body.get("claim") or ""),
        session_id=session.id,
        frame=frame_from_dict(body.get("frame"), session.camera_consent),
        audio=audio_from_dict(body.get("audio"), session.mic_consent),
        distress=int(distress) if distress is not None else None,
    )
    return outcome.public()


@route("POST", r"/api/patient/sessions/([^/]+)/end", (Role.PATIENT,))
def end_session(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _patient(ctx, user)
    session = ctx.repo.get_session(params[0])
    if not session or session.patient_id != patient.id:
        raise ValueError("unknown session")
    if session.status != "open":
        return {"session": session.public()}

    expired = ctx.engine.session_expired(session)
    distress = body.get("distress_end")
    session = ctx.engine.close_session(
        session.id,
        # An idle session closes with the last real rating, not one back-filled now.
        None if expired else (int(distress) if distress is not None else None),
        reason="timeout" if expired else "patient",
    )
    ctx.repo.audit(
        user.id, "session.closed", session.id, {"reason": "timeout" if expired else "patient"}
    )
    return {"session": session.public()}


@route("GET", r"/api/patient/medications", (Role.PATIENT,))
def patient_medications(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _patient(ctx, user)
    return {
        "medications": [med.public() for med in ctx.repo.list_medications(patient.id)],
        "adherence": medications.adherence(ctx.repo, patient.id),
    }


@route("GET", r"/api/patient/doses", (Role.PATIENT,))
def patient_doses(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _patient(ctx, user)
    doses = medications.ensure_doses(ctx.repo, patient.id)
    return {"doses": [dose.public() for dose in doses]}


@route("POST", r"/api/patient/doses/([^/]+)", (Role.PATIENT,))
def act_on_dose(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _patient(ctx, user)
    dose = ctx.repo.get_dose(params[0])
    if not dose or dose.patient_id != patient.id:
        raise ValueError("unknown dose")
    updated = medications.mark_dose(ctx.repo, params[0], str(body.get("action", "")), user.id)
    return {"dose": updated.public(), "adherence": medications.adherence(ctx.repo, patient.id)}


@route("GET", r"/api/patient/reminders", (Role.PATIENT,))
def patient_reminders(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _patient(ctx, user)
    return medications.due_reminders(
        ctx.repo, patient.id, tz_offset_minutes=int(body.get("tz_offset_minutes") or 0)
    )


# ------------------------------------------------------------------ caretaker
@route("GET", r"/api/caretaker/dashboard", (Role.CARETAKER,))
def caretaker_dashboard(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    return progress.caretaker_dashboard(ctx.repo, user.id)


@route("GET", r"/api/caretaker/patients", (Role.CARETAKER,))
def caretaker_patients(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    return {"patients": [p.public() for p in ctx.repo.patients_for_user(user.id, Role.CARETAKER)]}


@route("GET", r"/api/caretaker/patients/([^/]+)/progress", (Role.CARETAKER,))
def caretaker_progress(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _guard(ctx, user, params[0])
    return progress.patient_progress(ctx.repo, patient.id)


@route("GET", r"/api/caretaker/inbox", (Role.CARETAKER,))
def caretaker_inbox(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    return {"reports": report_service.inbox(ctx.repo, user.id)}


@route("POST", r"/api/caretaker/reports/([^/]+)/ack", (Role.CARETAKER,))
def caretaker_ack(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    return {"acknowledged": report_service.acknowledge(ctx.repo, params[0], user.id)}


# ------------------------------------------------------------------- therapist
@route("GET", r"/api/therapist/dashboard", (Role.THERAPIST,))
def therapist_dashboard(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    return progress.therapist_dashboard(ctx.repo, user.id)


@route("GET", r"/api/therapist/patients", (Role.THERAPIST,))
def therapist_patients(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    return {"patients": [p.public() for p in ctx.repo.patients_for_user(user.id, Role.THERAPIST)]}


@route("GET", r"/api/therapist/patients/([^/]+)/progress", (Role.THERAPIST,))
def therapist_progress(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _guard(ctx, user, params[0])
    return progress.patient_progress(ctx.repo, patient.id)


@route("GET", r"/api/therapist/patients/([^/]+)/reports", (Role.THERAPIST,))
def therapist_reports(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _guard(ctx, user, params[0])
    return {"reports": [r.public() for r in ctx.repo.list_reports_for_patient(patient.id)]}


@route("GET", r"/api/therapist/inbox", (Role.THERAPIST,))
def therapist_inbox(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    return {"reports": report_service.inbox(ctx.repo, user.id)}


@route("POST", r"/api/therapist/reports/([^/]+)/ack", (Role.THERAPIST,))
def therapist_ack(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    return {"acknowledged": report_service.acknowledge(ctx.repo, params[0], user.id)}


@route("GET", r"/api/therapist/patients/([^/]+)/notes", (Role.THERAPIST,))
def therapist_notes(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _guard(ctx, user, params[0])
    return {"notes": [note.public() for note in ctx.repo.list_notes(patient.id)]}


@route("POST", r"/api/therapist/patients/([^/]+)/notes", (Role.THERAPIST,))
def add_note(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _guard(ctx, user, params[0])
    text = str(body.get("body") or "").strip()
    if len(text) < 3:
        raise ValueError("note body is too short")
    note = Note(
        id=new_id("note"),
        patient_id=patient.id,
        author_id=user.id,
        author_role=Role.THERAPIST,
        author_name=user.display_name,
        body=text,
        created_at=utcnow(),
        pinned=bool(body.get("pinned")),
    )
    ctx.repo.add_note(note)
    ctx.repo.audit(user.id, "note.created", note.id, {"patient_id": patient.id})
    report_service.create_report(
        ctx.repo,
        patient_id=patient.id,
        kind="therapist_note",
        title=f"Therapist note — {user.display_name}",
        summary=text[:400],
        payload=note.public(),
        author_id=user.id,
        author_role=Role.THERAPIST,
        gateway=ctx.gateway,
    )
    return {"note": note.public()}


@route("GET", r"/api/therapist/patients/([^/]+)/medications", (Role.THERAPIST,))
def therapist_medications(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _guard(ctx, user, params[0])
    return {
        "medications": [med.public() for med in ctx.repo.list_medications(patient.id)],
        "adherence": medications.adherence(ctx.repo, patient.id),
    }


@route("POST", r"/api/therapist/patients/([^/]+)/medications", (Role.THERAPIST,))
def add_medication(ctx: AppContext, params, body, user, token) -> Dict[str, Any]:
    patient = _guard(ctx, user, params[0])
    times = body.get("times") or []
    if not isinstance(times, list) or not times:
        raise ValueError("times must be a non-empty list of HH:MM strings")
    med = Medication(
        id=new_id("med"),
        patient_id=patient.id,
        name=str(body.get("name") or "").strip(),
        dose=str(body.get("dose") or "").strip(),
        times=[str(t) for t in times],
        instructions=str(body.get("instructions") or ""),
        created_at=utcnow(),
    )
    if not med.name or not med.dose:
        raise ValueError("name and dose are required")
    for hhmm in med.times:
        medications._parse_time(hhmm)
    ctx.repo.add_medication(med)
    ctx.repo.audit(user.id, "medication.created", med.id, med.public())
    medications.ensure_doses(ctx.repo, patient.id)
    return {"medication": med.public()}


# ---------------------------------------------------------------------- server
class ShezoHandler(BaseHTTPRequestHandler):
    server_version = "ShezoAPI/0.1"
    ctx: AppContext

    def log_message(self, fmt: str, *args: Any) -> None:  # keep test output clean
        if os.environ.get("SHEZO_API_LOG"):
            super().log_message(fmt, *args)

    # -- plumbing
    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    # -- static front-end
    def _static(self, path: str) -> None:
        root = WEB_ROOT.resolve()
        relative = path.lstrip("/") or "index.html"
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            self._send(403, {"ok": False, "error": "forbidden"})
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            body = (
                "<!doctype html><meta charset='utf-8'><title>Shezo</title>"
                "<body style='font-family:system-ui;background:#0b1220;color:#e2e8f0;padding:3rem'>"
                "<h1>Shezo web app not found</h1>"
                "<p>The front-end lives in <code>web/</code> next to the <code>shezo</code> package.</p>"
                "<p>The JSON API is still running: <code>GET /api/health</code>.</p></body>"
            ).encode("utf-8")
            self.send_response(404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        raw = target.read_bytes()
        suffix = target.suffix.lower()
        cache = "no-store" if suffix == ".html" else "public, max-age=300"
        self.send_response(200)
        self.send_header("Content-Type", MIME_TYPES.get(suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def _token(self) -> Optional[str]:
        header = self.headers.get("Authorization") or ""
        if header.lower().startswith("bearer "):
            return header[7:].strip()
        return None

    def _dispatch(self, method: str) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            body = self._body()
            for route_method, pattern, handler, roles in ROUTES:
                if route_method != method:
                    continue
                match = pattern.fullmatch(path)
                if not match:
                    continue
                token = self._token()
                user: Optional[User] = None
                if roles:
                    user = self.ctx.auth.require_role(token, *roles)
                elif token:
                    user = self.ctx.repo.token_user(token)
                result = handler(self.ctx, match.groups(), body, user, token)
                self._send(200, {"ok": True, **result})
                return
            self._send(404, {"ok": False, "error": f"no route for {method} {path}"})
        except AuthError as exc:
            self._send(401, {"ok": False, "error": str(exc)})
        except RoleError as exc:
            self._send(403, {"ok": False, "error": str(exc)})
        except PermissionError as exc:
            self._send(403, {"ok": False, "error": str(exc)})
        except (ValueError, KeyError) as exc:
            self._send(400, {"ok": False, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send(500, {"ok": False, "error": f"internal error: {type(exc).__name__}"})

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path.startswith("/api/"):
            self._dispatch("GET")
        else:
            try:
                self._static(urlparse(self.path).path)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._send(405, {"ok": False, "error": "method not allowed"})

    def do_PUT(self) -> None:  # noqa: N802
        self._send(405, {"ok": False, "error": "method not allowed"})


def make_handler(ctx: AppContext) -> type:
    return type("BoundShezoHandler", (ShezoHandler,), {"ctx": ctx})


def serve(
    ctx: AppContext, host: str = "127.0.0.1", port: int = 8080
) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(ctx))
    httpd.daemon_threads = True
    return httpd


def main() -> None:
    host = os.environ.get("SHEZO_HOST", "127.0.0.1")
    # PaaS hosts (Render and friends) inject PORT; SHEZO_PORT wins locally.
    port = int(os.environ.get("SHEZO_PORT") or os.environ.get("PORT") or "8080")
    repo, label = open_backend()
    gateway = build_gateway()
    ctx = AppContext(
        repo=repo,
        auth=AuthService(repo),
        engine=GroundingEngine(repo, llm=None, gateway=gateway),
        gateway=gateway,
    )
    httpd = serve(ctx, host, port)
    print(f"Shezo on http://{host}:{httpd.server_address[1]}  (storage={label})")
    print("Web app: /   ·   API: /api/health   ·   seed demo data: py -m shezo.seed")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
        ctx.repo.close()


if __name__ == "__main__":
    main()
