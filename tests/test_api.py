"""End-to-end HTTP behaviour: role guards, consent, routing, notes."""
from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request

from shezo.api import serve
from tests.helpers import build_world


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = build_world()
        cls.ctx = cls.world["ctx"]
        cls.httpd = serve(cls.ctx, "127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        cls.ctx.repo.close()

    def call(self, method, path, body=None, token=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def login(self, email, password):
        status, payload = self.call("POST", "/api/auth/login", {"email": email, "password": password})
        self.assertEqual(status, 200, payload)
        return payload["token"]

    def test_health_is_public(self):
        status, payload = self.call("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")

    def test_protected_route_requires_token(self):
        status, payload = self.call("GET", "/api/patient/home")
        self.assertEqual(status, 401)
        self.assertFalse(payload["ok"])

    def test_full_flow_across_three_interfaces(self):
        patient_token = self.login("p@home.example", self.world["passwords"]["patient"])
        therapist_token = self.login("t@clinic.example", self.world["passwords"]["therapist"])
        caretaker_token = self.login("c@home.example", self.world["passwords"]["caretaker"])

        # Wrong role is refused.
        status, _ = self.call("GET", "/api/therapist/dashboard", token=patient_token)
        self.assertEqual(status, 403)
        status, _ = self.call("GET", "/api/patient/home", token=therapist_token)
        self.assertEqual(status, 403)

        status, session_payload = self.call(
            "POST",
            "/api/patient/sessions",
            {"distress_start": 8, "camera_consent": True, "mic_consent": True},
            token=patient_token,
        )
        self.assertEqual(status, 200, session_payload)
        session_id = session_payload["session"]["id"]

        status, check_payload = self.call(
            "POST",
            f"/api/patient/sessions/{session_id}/checks",
            {
                "claim": "There is a man standing in the corner watching me.",
                "distress": 8,
                "frame": {"labels": {"chair": 0.7}, "brightness": 0.6, "blur": 0.1},
                "audio": {"speech_detected": False, "ambient_db": 30},
            },
            token=patient_token,
        )
        self.assertEqual(status, 200, check_payload)
        self.assertEqual(check_payload["reality_check"]["verdict"], "no_sensor_evidence")
        self.assertTrue(check_payload["reality_check"]["disclosure"])

        # Both care-team interfaces received the report.
        status, therapist_inbox = self.call("GET", "/api/therapist/inbox", token=therapist_token)
        self.assertEqual(status, 200)
        self.assertTrue(therapist_inbox["reports"])
        report = therapist_inbox["reports"][0]
        self.assertEqual(report["kind"], "reality_check")

        status, caretaker_inbox = self.call("GET", "/api/caretaker/inbox", token=caretaker_token)
        self.assertEqual(status, 200)
        self.assertTrue(caretaker_inbox["reports"])

        status, ack = self.call(
            "POST", f"/api/therapist/reports/{report['id']}/ack", token=therapist_token
        )
        self.assertEqual(status, 200)
        self.assertTrue(ack["acknowledged"])

        # Therapist notes land on the shared record and in the caretaker inbox.
        status, note_payload = self.call(
            "POST",
            f"/api/therapist/patients/{self.world['patient'].id}/notes",
            {"body": "Reviewed the corner episode; plan: practice 5-4-3-2-1 nightly."},
            token=therapist_token,
        )
        self.assertEqual(status, 200, note_payload)

        status, dashboard = self.call("GET", "/api/therapist/dashboard", token=therapist_token)
        self.assertEqual(status, 200)
        self.assertEqual(len(dashboard["patients"]), 1)
        self.assertIn("trend", dashboard["patients"][0])

        status, caretaker_dash = self.call("GET", "/api/caretaker/dashboard", token=caretaker_token)
        self.assertEqual(status, 200)
        self.assertEqual(caretaker_dash["summary"]["patients"], 1)

        status, home = self.call("GET", "/api/patient/home", token=patient_token)
        self.assertEqual(status, 200)
        self.assertIn("reminders", home)
        # A reload finds the same open session instead of losing it.
        self.assertEqual(home["open_session"]["id"], session_id)

        status, reopened = self.call(
            "POST", "/api/patient/sessions", {"distress_start": 6}, token=patient_token
        )
        self.assertEqual(status, 200, reopened)
        self.assertTrue(reopened["resumed"])
        self.assertEqual(reopened["session"]["id"], session_id)

    def test_unknown_patient_is_forbidden_not_leaked(self):
        token = self.login("t@clinic.example", self.world["passwords"]["therapist"])
        status, _ = self.call("GET", "/api/therapist/patients/pat_does_not_exist/progress", token=token)
        self.assertEqual(status, 403)

    def test_check_without_consent_ignores_camera(self):
        token = self.login("p@home.example", self.world["passwords"]["patient"])
        status, payload = self.call(
            "POST", "/api/patient/sessions", {"distress_start": 4}, token=token
        )
        session_id = payload["session"]["id"]
        status, check_payload = self.call(
            "POST",
            f"/api/patient/sessions/{session_id}/checks",
            {
                "claim": "There is a person in the room.",
                "frame": {"labels": {"person": 0.99}, "brightness": 0.9, "blur": 0.0},
            },
            token=token,
        )
        self.assertEqual(status, 200)
        self.assertEqual(check_payload["reality_check"]["verdict"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
