"""The deterministic safety floor must never be softened by later stages."""
from __future__ import annotations

import unittest

from shezo.ai.safety import DISCLOSURE, screen
from shezo.models import RiskLevel


class SafetyTests(unittest.TestCase):
    def test_ordinary_message_is_low_risk(self):
        result = screen("I feel shaky but I know you are checking with me.", distress=2)
        self.assertIs(result.risk_level, RiskLevel.LOW)
        self.assertFalse(result.crisis_protocol)

    def test_self_harm_is_critical_and_offers_resources(self):
        result = screen("I want to kill myself tonight.")
        self.assertIs(result.risk_level, RiskLevel.CRITICAL)
        self.assertTrue(result.crisis_protocol)
        self.assertTrue(result.notify_emergency_contact)
        self.assertIn("self_harm_language", result.flags)
        self.assertIn("988", result.note)

    def test_harm_to_others_is_critical(self):
        result = screen("The voice says I should hurt him.")
        self.assertIs(result.risk_level, RiskLevel.CRITICAL)
        self.assertIn("harm_to_others_language", result.flags)

    def test_command_hallucination_is_high(self):
        result = screen("A voice tells me to leave the house right now.")
        self.assertIs(result.risk_level, RiskLevel.HIGH)
        self.assertIn("command_hallucination", result.flags)
        self.assertFalse(result.crisis_protocol)

    def test_extreme_distress_is_critical(self):
        result = screen("It is very loud in my head.", distress=9)
        self.assertIs(result.risk_level, RiskLevel.CRITICAL)
        self.assertIn("extreme_distress", result.flags)

    def test_stopping_medication_is_elevated(self):
        result = screen("I stopped taking my meds last week.")
        self.assertIs(result.risk_level, RiskLevel.ELEVATED)

    def test_disclosure_forbids_definitive_claims(self):
        self.assertIn("does not diagnose", DISCLOSURE)
        self.assertIn("cannot confirm", DISCLOSURE)


if __name__ == "__main__":
    unittest.main()
