# Shezo — AI grounding model for schizophrenia support

Python model + JSON API for an AI-assisted grounding platform. One shared patient
record, three role-scoped interfaces (patient, caretaker, therapist), camera/mic
reality checks, medication reminders, and therapist-supervised progress tracking.

The repository contains the model, the rules engine, the JSON API and a plain
HTML/CSS/JS web app (`web/`, Tailwind + Chart.js) with the three role-scoped
dashboards. The UI does not change the AI contract: the browser sends summarised
sensor numbers, never video or audio.

---

## What it does

| Capability | Where |
| --- | --- |
| Camera/mic reality check ("is this real?") | `shezo/ai/engine.py` |
| Claim parsing (what was reported, visual/auditory, command?) | `shezo/ai/claims.py` |
| Sensor-evidence verdict + confidence | `shezo/ai/scoring.py` |
| Deterministic crisis screening (cannot be overridden) | `shezo/ai/safety.py` |
| Narration, optional LLM, offline fallback | `shezo/ai/llm.py` |
| Grounding exercises (5-4-3-2-1, breathing, contact a human) | `shezo/ai/grounding.py` |
| Shared patient record + audit trail | `shezo/storage.py` |
| Reports routed to assigned therapist + caretakers | `shezo/services/reports.py` |
| Medication schedule, reminders, adherence, missed-dose alerts | `shezo/services/medications.py` |
| Progress / improvement aggregation and dashboards | `shezo/services/progress.py` |
| Role-scoped HTTP API | `shezo/api.py` |

## Safety posture (read this before changing the model)

This is **decision support, not a medical device**, and it is built so that it
cannot do the most harmful thing a "reality checker" could do:

1. **It never says an experience is not real.** A verdict describes *sensor
   evidence only* — `corroborated`, `inconclusive`, or `no_sensor_evidence`.
   Confidence is capped at 0.90 and the disclosure sentence travels with every
   check.
2. **It never diagnoses.** No output contains "you are hallucinating/delusional".
   The optional LLM cannot set the verdict, confidence or risk; it only writes
   prose, and its output is rejected if it contains a definitive claim.
3. **The safety floor is deterministic and runs first.** `ai/safety.py` screens
   self-harm, harm to others, command hallucinations, severe disorganisation and
   extreme distress. Later stages may *raise* risk, never lower it.
4. **Command hallucinations always escalate.** Any "the voices tell me to…"
   automatically produces a HIGH-risk, urgent report to the care team, plus the
   "do not act for 10 minutes" and "contact a human" steps.
5. **Humans are always in the loop.** Every check produces a report to the
   assigned therapist and caretakers. A crisis check additionally alerts the
   emergency contact path.
6. **Consent gates the sensors.** Camera/mic samples are dropped unless an open
   session recorded consent for that device. No raw frames or audio are stored —
   only device-extracted labels and levels.

## Quickstart

Requires Python 3.10+ (tested on 3.15). The model and API use the standard
library only; the MySQL backend additionally needs PyMySQL.

```bash
py -m pip install pymysql        # only for the MySQL/MariaDB backend
py -m shezo.seed                 # demo accounts + two weeks of history
py -m shezo.api                  # web app + API on http://127.0.0.1:8080

py -m shezo.demo                 # model pipeline only, in-memory, prints JSON
py -m unittest discover -s tests -t .   # test suite
```

Then open <http://127.0.0.1:8080/> and sign in with an account printed by the
seed (`jordan@example.com` / `patient-pass-1`, `sam@family.example` /
`family-pass-1`, `dr.reyes@clinic.example` / `therapy-pass-1`).

The demo seeds a therapist, caretaker and patient, then runs: a visual report the
camera cannot confirm → a report the camera *does* corroborate → a command
hallucination that escalates → medication taken/missed → both care-team
dashboards.

## Web app

`web/` is one page served by the API process itself (`GET /`), so the browser and
the JSON API share an origin and no CORS or second dev server is needed.

| View | What it does |
| --- | --- |
| Patient | Opens a consented grounding session, captures sensor summaries, runs a reality check, shows the verdict, grounding steps and disclosure, handles due/missed doses, shows the week's trend and care team |
| Caretaker | Caseload with trend badges, adherence and distress charts, verdict mix, report inbox with acknowledgement |
| Therapist | Same progress detail plus the patient's report list, clinical notes (routed to the caretaker) and the medication plan |

**Camera and microphone.** The browser computes brightness, a Laplacian blur
estimate, frame-to-frame motion, face count (only where `FaceDetector` exists) and
loudness/speech presence with the Web Audio API, then posts those numbers. No
frames or audio leave the device and nothing is stored. Because a browser cannot
diarise speakers, `other_speaker` is always `false` from the web app — the model
therefore never claims the microphone heard a *second* voice.

**Dashboard charts.** Chart.js is vendored in `web/vendor/` and two small plugins
are registered in `web/app.js`: one badges the latest point of each line (weekly
adherence, in-session distress) and one draws the centre label of the verdict
doughnut.

Tailwind is compiled ahead of time, not pulled from a CDN:

```bash
cd web && npm install
npx tailwindcss -c tailwind.config.js -i src/input.css -o tailwind.css --minify
```

## Storage

`Repository` is written once against a small connection surface, so the same
queries run on either backend:

- **MySQL / MariaDB (default for the API)** — `shezo/mysql.py` adapts PyMySQL,
  creates the `shezo` database and tables on first run, and the tables appear in
  phpMyAdmin. Configure with `SHEZO_MYSQL_URL`
  (`mysql://user:password@host:3306/database`).
- **SQLite** — the default for the tests and `shezo.demo`, and the fallback the
  server uses (with a printed warning) if MySQL is unreachable.

`SHEZO_DB_ENGINE=mysql|sqlite` selects the backend for `py -m shezo.api` and
`py -m shezo.seed`.

## API

`POST /api/auth/login` → `{token}`; send `Authorization: Bearer <token>`.

### Patient
| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/patient/home` | reminders, recent checks, trend, care team, `open_session` |
| POST | `/api/patient/sessions` | open a grounding session (consent flags); resumes an open one |
| POST | `/api/patient/sessions/{id}/checks` | run a reality check |
| POST | `/api/patient/sessions/{id}/end` | close session, write summary report |
| GET | `/api/patient/medications` | plan + adherence |
| GET | `/api/patient/doses` | today's doses (materialised on demand) |
| POST | `/api/patient/doses/{id}` | `{"action": "taken" \| "skipped"}` |
| GET | `/api/patient/reminders` | reminder card payload |
| GET | `/api/patient/reports` | reports shared with the care team |

A check body:

```json
{
  "claim": "There is a man standing in the corner watching me.",
  "distress": 8,
  "frame": {"labels": {"chair": 0.7}, "brightness": 0.6, "blur": 0.1,
            "faces": 0, "motion": 0.02, "seconds_since_frame": 0.1},
  "audio": {"speech_detected": false, "other_speaker": false,
            "ambient_db": 30, "duration_s": 3.0, "snr_db": 14}
}
```

Response (trimmed):

```json
{
  "reality_check": {
    "verdict": "no_sensor_evidence",
    "confidence": 0.72,
    "headline": "The camera was on, and it didn't pick that up this time.",
    "explanation": "...",
    "grounding_steps": [{"title": "5-4-3-2-1 anchoring", "instructions": ["..."]}],
    "disclosure": "Shezo reports what the camera and microphone did or did not pick up..."
  },
  "safety": {"risk_level": "high", "flags": ["high_distress"], "crisis_protocol": false},
  "reports": [{"kind": "reality_check", "urgent": true,
               "recipients": [{"role": "therapist", "status": "recorded"},
                              {"role": "caretaker", "status": "recorded"}]}],
  "narration_source": "offline"
}
```

### Caretaker
`GET /api/caretaker/dashboard`, `GET /api/caretaker/patients`,
`GET /api/caretaker/patients/{id}/progress`, `GET /api/caretaker/inbox`,
`POST /api/caretaker/reports/{id}/ack`.

### Therapist
`GET /api/therapist/dashboard`, `GET /api/therapist/patients`,
`GET /api/therapist/patients/{id}/progress`,
`GET /api/therapist/patients/{id}/reports`, `GET /api/therapist/inbox`,
`POST /api/therapist/reports/{id}/ack`,
`GET|POST /api/therapist/patients/{id}/notes`,
`POST /api/therapist/patients/{id}/medications`.

Role guards return 401/403. A therapist or caretaker can only reach patients
assigned to them in the care team; unknown ids are 403, not 404, so ids do not leak.

## Plugging in real models

**Vision/audio.** The engine consumes *samples*, not pixels. Implement
`shezo.ai.perception.FrameAnalyzer` / `AudioAnalyzer` and pass them to
`GroundingEngine(frame_analyzer=..., audio_analyzer=...)`; anything from an
on-device YOLO/CLIP detector to a cloud endpoint works. The default
`MetadataFrameAnalyzer` trusts your device detector's labels and refuses to
testify when the frame is dark, blurry, stale or moving too fast (it returns
`inconclusive` rather than a false "nothing there").

**Narration.** Set `SHEZO_LLM_API_KEY` (or `OPENAI_API_KEY`) plus optional
`SHEZO_LLM_BASE_URL`, `SHEZO_LLM_MODEL` for any OpenAI-compatible endpoint. With
no key the deterministic narrator is used — the demo and tests run fully offline.
An LLM failure, timeout, malformed response or unsafe phrase silently falls back.

**Delivery.** Reports are always written in-app and recorded in `deliveries`. Set
`SHEZO_WEBHOOK_URL` to also POST a JSON envelope to a relay (email/SMS/push).

## Data model

`users`, `auth_tokens`, `patients`, `care_team`, `medications`, `doses`,
`sessions`, `reality_checks`, `reports`, `report_recipients`, `deliveries`,
`notes`, `audit` (SQLite via `Repository(":memory:")`/a file path, or MySQL via
`Repository(dsn="mysql://…")`).

Doses are materialised per day per scheduled time, so adherence is a query.
A dose pending past the grace window (60 min default) becomes `missed` exactly
once and emits exactly one care-team alert.

## Environment

| Variable | Meaning |
| --- | --- |
| `SHEZO_DB_ENGINE` | `mysql` (default) or `sqlite` |
| `SHEZO_MYSQL_URL` | MySQL DSN, default `mysql://root@127.0.0.1:3306/shezo` |
| `SHEZO_DB` | SQLite path used when the engine is sqlite (default `shezo.db`) |
| `SHEZO_HOST` / `SHEZO_PORT` | bind address (default `127.0.0.1:8080`) |
| `SHEZO_LLM_API_KEY` | enables LLM narration |
| `SHEZO_LLM_BASE_URL`, `SHEZO_LLM_MODEL` | endpoint/model override |
| `SHEZO_WEBHOOK_URL` | extra delivery channel |
| `SHEZO_SESSION_TIMEOUT_MINUTES` | idle session auto-close, default `90` (min 5) |

## Honest limitations

- **Not clinically validated.** No accuracy claim is made or implied; the verdict
  is sensor coverage, not truth. Do not let this replace clinical judgement.
- **The default vision analyzer does not see images.** It reasons over labels your
  device produces; ship a real on-device detector before calling it "camera AI".
- **Auth is minimal** (scrypt + opaque bearer tokens, SQLite). Before multi-tenant
  deployment: TLS, a real IdP, token rotation, encrypted database, per-role key
  management, and a DPIA.
- **Single-node.** SQLite and an in-process scheduler; a real deployment needs a
  job runner for reminders and a durable queue for delivery.
- **No push notifications, no streaming media, coarse browser sensing.** The web
  app summarises frames and loudness locally; there is no object detector beyond
  the browser's `FaceDetector`, no speaker diarisation, and no video/audio
  telemetry. HIPAA/GDPR paperwork and a real delivery provider are still needed.
- **The web app is a local tool.** It binds to `127.0.0.1`, stores the bearer token
  in `sessionStorage`, and has no TLS, no CSRF token and no multi-tenant
  isolation beyond the care-team checks in the API.
- Crisis lines in `ai/safety.py` are examples (US 988, UK 111, EU 112); localise
  them per deployment.
