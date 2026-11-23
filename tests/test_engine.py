"""Reality-check pipeline: verdicts, consent gating, escalation, routing."""
from __future__ import annotations

import unittest

from shezo.ai.perception import AudioSample, FrameSample
from shezo.models import Modality, RiskLevel, Role, Verdict
from tests.helpers import build_world

CLEAR_ROOM = FrameSample(labels={"chair": 0.6, "door": 0.9}, brightness=0.6, blur=0.1)
DARK_ROOM = FrameSample(labels={}, brightness=0.02, blur=0.9)
QUIET_MIC = AudioSample(duration_s=3.0, ambient_db=30.0, speech_detected=False)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.world = build_world()
        self.repo = self.world["repo"]
        self.engine = self.world["engine"]
        self.patient = self.world["patient"]
        self.session = self.engine.open_session(
            self.patient.id, distress=7, camera_consent=True, mic_consent=True
        )

    def tearDown(self):
        self.repo.close()

    def test_visual_report_without_matching_detection(self):
        outcome = self.engine.run_reality_check(
            self.patient.id,
            "There is a man standing in the corner watching me.",
            session_id=self.session.id,
            frame=CLEAR_ROOM,
            audio=QUIET_MIC,
            distress=4,
        )
        check = outcome.check
        self.assertIs(check.verdict, Verdict.NO_SENSOR_EVIDENCE)
        self.assertIs(check.modality, Modality.VISUAL)
        self.assertLessEqual(check.confidence, 0.9)
        self.assertFalse(check.escalated)
        self.assertTrue(check.disclosure)
        self.assertTrue(check.grounding_steps)

    def test_narration_never_declares_the_experience_false(self):
        outcome = self.engine.run_reality_check(
            self.patient.id,
            "There is a shadow figure by the window.",
            session_id=self.session.id,
            frame=CLEAR_ROOM,
            audio=QUIET_MIC,
        )
        text = f"{outcome.check.headline} {outcome.check.explanation}".lower()
        for forbidden in ("you are hallucinating", "it is not real", "you are delusional", "you are psychotic"):
            self.assertNotIn(forbidden, text)
        self.assertIn("didn't pick", text)

    def test_person_detected_is_corroborated(self):
        outcome = self.engine.run_reality_check(
            self.patient.id,
            "Is there someone at the door? I think I see a person.",
            session_id=self.session.id,
            frame=FrameSample(labels={"person": 0.94}, brightness=0.7, blur=0.05),
            audio=QUIET_MIC,
        )
        self.assertIs(outcome.check.verdict, Verdict.CORROBORATED)
        self.assertGreater(outcome.check.confidence, 0.6)

    def test_no_camera_means_inconclusive(self):
        outcome = self.engine.run_reality_check(
            self.patient.id,
            "There is someone behind me.",
            session_id=self.session.id,
            frame=None,
            audio=None,
        )
        self.assertIs(outcome.check.verdict, Verdict.INCONCLUSIVE)
        self.assertLessEqual(outcome.check.confidence, 0.4)

    def test_unusable_frame_is_inconclusive_not_a_denial(self):
        outcome = self.engine.run_reality_check(
            self.patient.id,
            "There is someone behind me.",
            session_id=self.session.id,
            frame=DARK_ROOM,
            audio=QUIET_MIC,
        )
        self.assertIs(outcome.check.verdict, Verdict.INCONCLUSIVE)

    def test_consent_gate_drops_camera_when_not_granted(self):
        session = self.engine.open_session(
            self.patient.id, distress=5, camera_consent=False, mic_consent=False
        )
        outcome = self.engine.run_reality_check(
            self.patient.id,
            "There is a person in the room.",
            session_id=session.id,
            frame=FrameSample(labels={"person": 0.99}, brightness=0.9, blur=0.0),
            audio=AudioSample(speech_detected=True, other_speaker=True),
        )
        self.assertIs(outcome.check.verdict, Verdict.INCONCLUSIVE)

    def test_command_hallucination_escalates_and_routes(self):
        outcome = self.engine.run_reality_check(
            self.patient.id,
            "The voices tell me to go outside and meet them.",
            session_id=self.session.id,
            frame=CLEAR_ROOM,
            audio=QUIET_MIC,
            distress=9,
        )
        self.assertTrue(outcome.check.escalated)
        self.assertIs(outcome.check.risk_level, RiskLevel.CRITICAL)
        kinds = {report.kind.value for report in outcome.reports}
        self.assertIn("reality_check", kinds)
        self.assertIn("crisis_escalation", kinds)
        self.assertTrue(any(report.urgent for report in outcome.reports))

    def test_reports_go_to_therapist_and_caretaker(self):
        outcome = self.engine.run_reality_check(
            self.patient.id,
            "I hear a voice whispering my name.",
            session_id=self.session.id,
            audio=QUIET_MIC,
            frame=CLEAR_ROOM,
        )
        report = outcome.reports[0]
        roles = {recipient["role"] for recipient in report.recipients}
        self.assertEqual(roles, {Role.THERAPIST.value, Role.CARETAKER.value})
        self.assertEqual(len(report.recipients), 2)
        self.assertTrue(all(r["status"] == "recorded" for r in report.recipients))
        delivered = self.repo.deliveries_for(report.id)
        self.assertEqual(len(delivered), 2)

    def test_auditory_report_cannot_be_confirmed_by_mic(self):
        outcome = self.engine.run_reality_check(
            self.patient.id,
            "I hear people talking about me but I can't find them.",
            session_id=self.session.id,
            audio=QUIET_MIC,
        )
        self.assertIs(outcome.check.verdict, Verdict.NO_SENSOR_EVIDENCE)
        self.assertIn("cannot be recorded", outcome.check.sensor_summary + outcome.check.explanation)

    def test_closing_session_produces_summary(self):
        self.engine.run_reality_check(
            self.patient.id,
            "There is a figure in the hallway.",
            session_id=self.session.id,
            frame=CLEAR_ROOM,
        )
        closed = self.engine.close_session(self.session.id, distress_end=3)
        self.assertEqual(closed.status, "closed")
        kinds = [
            report.kind.value for report in self.repo.list_reports_for_patient(self.patient.id)
        ]
        self.assertIn("session_summary", kinds)

    def test_resume_returns_the_session_that_is_still_open(self):
        resumed = self.engine.resume_session(self.patient.id)
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.id, self.session.id)

    def test_idle_session_is_closed_and_summarised(self):
        self.engine.close_session(self.session.id, distress_end=3)
        stale = self.engine.open_session(
            self.patient.id, distress=6, camera_consent=True, mic_consent=True
        )
        # Simulate a tab left open for days: no checks since it started.
        stale.started_at = "2020-01-01T00:00:00+00:00"
        self.repo.update_session(stale)

        self.assertIsNone(self.engine.resume_session(self.patient.id))
        stored = self.repo.get_session(stale.id)
        self.assertEqual(stored.status, "closed")
        self.assertIsNotNone(stored.ended_at)
        summaries = [
            report
            for report in self.repo.list_reports_for_patient(self.patient.id)
            if report.kind.value == "session_summary"
        ]
        self.assertEqual(len(summaries), 2)
        self.assertIn("automatically", summaries[0].summary)
        self.assertEqual(summaries[0].payload["closed_reason"], "timeout")

    def test_resumed_session_takes_the_new_consent(self):
        adopted = self.engine.adopt_session(
            self.session, distress=9, camera_consent=False, mic_consent=False
        )
        stored = self.repo.get_session(self.session.id)
        self.assertEqual(adopted.id, self.session.id)
        self.assertFalse(stored.camera_consent)
        self.assertFalse(stored.mic_consent)
        self.assertEqual(stored.distress_start, 7)

    def test_closed_session_rejects_new_checks(self):
        self.engine.close_session(self.session.id, distress_end=2)
        with self.assertRaises(ValueError):
            self.engine.run_reality_check(
                self.patient.id, "anything", session_id=self.session.id
            )


if __name__ == "__main__":
    unittest.main()
