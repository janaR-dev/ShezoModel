"""Shezo — AI-assisted grounding for people who experience psychosis.

Three role-scoped interfaces share one patient record:

* patient   — grounding sessions (camera/mic reality checks), medication reminders
* caretaker — improvement view and alerts
* therapist — reports, notes, progress dashboard

The Python package has no third-party dependencies; see README.md.
"""

__version__ = "0.1.0"

from .models import (  # noqa: F401
    DoseStatus,
    Modality,
    Patient,
    ReportKind,
    RiskLevel,
    Role,
    Session,
    Verdict,
)
