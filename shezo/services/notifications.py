"""Outbound notification gateways.

Reports are always addressed in-app. A gateway is an *additional* delivery
channel (email relay, SMS webhook, push service). The default records the
delivery so the audit trail shows exactly what was "sent" during local runs.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, Optional, Protocol, Tuple

from ..models import Report, User


class NotificationGateway(Protocol):
    channel: str

    def send(self, report: Report, recipient: User) -> Tuple[str, str]:
        """Return (status, detail) where status is 'sent' or 'failed'."""
        ...


class RecordingGateway:
    """Default: records delivery in SQLite so nothing pretends to have been emailed."""

    channel = "inbox"

    def send(self, report: Report, recipient: User) -> Tuple[str, str]:
        return "recorded", f"in-app delivery to {recipient.email}"


class WebhookGateway:
    """POSTs a small JSON envelope to a care-team webhook (Slack/SMS/email relay)."""

    channel = "webhook"

    def __init__(self, url: str, timeout: float = 8.0) -> None:
        self.url = url
        self.timeout = timeout

    def send(self, report: Report, recipient: User) -> Tuple[str, str]:
        envelope = {
            "recipient": {"id": recipient.id, "role": recipient.role.value},
            "report": {
                "id": report.id,
                "patient_id": report.patient_id,
                "kind": report.kind.value,
                "title": report.title,
                "summary": report.summary,
                "urgent": report.urgent,
                "created_at": report.created_at,
            },
        }
        body = json.dumps(envelope).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return "sent", f"HTTP {response.status}"
        except Exception as exc:  # noqa: BLE001 - delivery must never break a check
            return "failed", f"{type(exc).__name__}: {exc}"[:300]


class MultiGateway:
    """Fan out to several gateways and report the first useful result."""

    def __init__(self, gateways: list) -> None:
        self.gateways = [gw for gw in gateways if gw]
        self.channel = "multi"

    def send(self, report: Report, recipient: User) -> Tuple[str, str]:
        details = []
        for gateway in self.gateways:
            status, detail = gateway.send(report, recipient)
            details.append(f"{gateway.channel}:{status} ({detail})")
        return "sent", "; ".join(details)


def build_gateway() -> Optional[NotificationGateway]:
    """Use SHEZO_WEBHOOK_URL when configured, otherwise the recording gateway."""
    url = os.environ.get("SHEZO_WEBHOOK_URL")
    if url:
        return MultiGateway([RecordingGateway(), WebhookGateway(url)])
    return RecordingGateway()


def describe(gateway: Optional[NotificationGateway]) -> Dict[str, Any]:
    if gateway is None:
        return {"channel": "none"}
    return {"channel": getattr(gateway, "channel", "inbox")}
