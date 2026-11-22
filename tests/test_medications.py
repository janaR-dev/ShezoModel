"""Medication scheduling, adherence and missed-dose escalation."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from shezo.models import DoseStatus
from shezo.services import medications
from tests.helpers import build_world


class MedicationTests(unittest.TestCase):
    def setUp(self):
        self.world = build_world()
        self.repo = self.world["repo"]
        self.patient = self.world["patient"]

    def tearDown(self):
        self.repo.close()

    def test_dose_generation_is_idempotent(self):
        first = medications.ensure_doses(self.repo, self.patient.id)
        self.assertEqual(len(first), 2)
        again = medications.ensure_doses(self.repo, self.patient.id)
        self.assertEqual(again, [])

    def test_taking_a_dose_updates_adherence(self):
        doses = medications.ensure_doses(self.repo, self.patient.id)
        updated = medications.mark_dose(self.repo, doses[0].id, "taken", self.world["patient_user"].id)
        self.assertIs(updated.status, DoseStatus.TAKEN)
        self.assertIsNotNone(updated.acted_at)

        stats = medications.adherence(self.repo, self.patient.id)
        self.assertEqual(stats["taken"], 1)
        self.assertEqual(stats["pending"], 1)
        self.assertEqual(stats["adherence_pct"], 100.0)

    def test_double_marking_is_rejected(self):
        doses = medications.ensure_doses(self.repo, self.patient.id)
        medications.mark_dose(self.repo, doses[0].id, "taken", "u")
        with self.assertRaises(ValueError):
            medications.mark_dose(self.repo, doses[0].id, "skipped", "u")

    def test_invalid_action_is_rejected(self):
        doses = medications.ensure_doses(self.repo, self.patient.id)
        with self.assertRaises(ValueError):
            medications.mark_dose(self.repo, doses[0].id, "maybe", "u")

    def test_missed_dose_alerts_care_team_once(self):
        now = datetime(2026, 1, 5, 7, 0, tzinfo=timezone.utc).replace(microsecond=0)
        doses = medications.ensure_doses(self.repo, self.patient.id, now=now.isoformat())
        self.assertEqual(len(doses), 2)

        # 12:00 is past the 09:00 dose + grace window; the 21:00 dose is still ahead.
        later = (now + timedelta(hours=5)).isoformat()
        missed = medications.refresh_missed(self.repo, self.patient.id, now=later)
        self.assertEqual(len(missed), 1)
        self.assertTrue(all(dose.status is DoseStatus.MISSED for dose in missed))
        self.assertEqual(missed[0].due_at[11:16], "09:00")

        reports = [
            report
            for report in self.repo.list_reports_for_patient(self.patient.id)
            if report.kind.value == "medication_missed"
        ]
        self.assertEqual(len(reports), 1)
        self.assertEqual(len(reports[0].recipients), 2)

        # A second sweep must not duplicate the alert.
        medications.refresh_missed(self.repo, self.patient.id, now=later)
        reports = [
            report
            for report in self.repo.list_reports_for_patient(self.patient.id)
            if report.kind.value == "medication_missed"
        ]
        self.assertEqual(len(reports), 1)

    def test_reminder_message_mentions_medication(self):
        from shezo.models import Dose, DoseStatus

        due = Dose(
            id="d1",
            medication_id="med_test",
            patient_id=self.patient.id,
            due_at="2026-01-05T09:00:00+00:00",
            status=DoseStatus.PENDING,
            medication_name="Risperidone",
            medication_dose="2 mg",
        )
        self.assertIn("Risperidone", medications._reminder_message([due], [], []))
        self.assertIn("not confirmed", medications._reminder_message([], [], [due]))

        reminders = medications.due_reminders(self.repo, self.patient.id)
        self.assertIn("due_now", reminders)
        self.assertIn("upcoming", reminders)
        self.assertIn("message", reminders)

    def test_bad_time_is_rejected(self):
        with self.assertRaises(ValueError):
            medications._parse_time("25:99")


if __name__ == "__main__":
    unittest.main()
