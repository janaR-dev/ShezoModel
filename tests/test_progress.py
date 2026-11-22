"""Progress aggregation and the role dashboards."""
from __future__ import annotations

import unittest

from shezo.ai.perception import AudioSample, FrameSample
from shezo.services import medications, progress
from tests.helpers import build_world


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.world = build_world()
        self.repo = self.world["repo"]
        self.engine = self.world["engine"]
        self.patient = self.world["patient"]

    def tearDown(self):
        self.repo.close()

    def _seed_session(self, start: int, end: int, claim: str = "There is a man in the corner."):
        session = self.engine.open_session(self.patient.id, distress=start, camera_consent=True)
        self.engine.run_reality_check(
            self.patient.id,
            claim,
            session_id=session.id,
            frame=FrameSample(labels={"chair": 0.6}, brightness=0.6, blur=0.1),
            audio=AudioSample(duration_s=3.0, speech_detected=False),
            distress=start,
        )
        self.engine.close_session(session.id, distress_end=end)
        return session

    def test_progress_counts_activity(self):
        self._seed_session(8, 4)
        self._seed_session(7, 3, claim="The voices are talking about me.")
        doses = medications.ensure_doses(self.repo, self.patient.id)
        medications.mark_dose(self.repo, doses[0].id, "taken", self.world["patient_user"].id)

        data = progress.patient_progress(self.repo, self.patient.id)
        self.assertEqual(data["sessions"]["count"], 2)
        self.assertEqual(data["sessions"]["avg_distress_start"], 7.5)
        self.assertEqual(data["sessions"]["avg_distress_end"], 3.5)
        self.assertEqual(data["sessions"]["avg_relief"], 4.0)
        # Each session produced one reality check plus one session summary.
        self.assertEqual(data["reality_checks"]["count"], 2)
        self.assertEqual(data["adherence"]["taken"], 1)
        self.assertEqual(data["trend"]["direction"], "baseline")
        self.assertGreaterEqual(len(data["weekly"]), 1)

    def test_trend_improving_when_distress_falls_and_adherence_rises(self):
        trend = progress._trend(
            avg_start=4.0,
            avg_end=2.0,
            prev_start=6.0,
            prev_end=6.0,
            adherence_pct=95.0,
            prev_adherence_pct=60.0,
            has_current=True,
            has_previous=True,
        )
        self.assertEqual(trend["direction"], "improving")
        self.assertGreater(trend["score"], 0)

    def test_trend_worsening(self):
        trend = progress._trend(
            avg_start=8.0,
            avg_end=9.0,
            prev_start=5.0,
            prev_end=5.0,
            adherence_pct=40.0,
            prev_adherence_pct=90.0,
            has_current=True,
            has_previous=True,
        )
        self.assertEqual(trend["direction"], "worsening")

    def test_caretaker_and_therapist_dashboards(self):
        self._seed_session(7, 3)
        caretaker_view = progress.caretaker_dashboard(self.repo, self.world["caretaker"].id)
        self.assertEqual(caretaker_view["summary"]["patients"], 1)
        self.assertTrue(caretaker_view["inbox"])

        therapist_view = progress.therapist_dashboard(self.repo, self.world["therapist"].id)
        self.assertEqual(len(therapist_view["patients"]), 1)
        self.assertIn("trend", therapist_view["patients"][0])

    def test_patient_home_exposes_reminders(self):
        home = progress.patient_home(self.repo, self.patient.id)
        self.assertIn("reminders", home)
        self.assertIn("care_team", home)
        self.assertEqual(len(home["care_team"]), 2)


if __name__ == "__main__":
    unittest.main()
