"""Claim parsing: the lexicon's vocabulary must survive the single-pass rewrite.

``parse_claim`` used to run one regex search per cue (~50 per check). It now runs
one alternation per cue family, which is faster but must stay behaviourally
identical, so these cases pin the modality, the entities and the command flag.
"""
from __future__ import annotations

import unittest

from shezo.ai.claims import parse_claim
from shezo.models import Modality


class ClaimParsingTests(unittest.TestCase):
    def test_modality_and_entities(self):
        cases = [
            ("There is a man standing in the corner watching me.", Modality.VISUAL, ["person"]),
            ("I can hear whispering coming from the wall.", Modality.AUDITORY, []),
            ("I see a spider and I hear buzzing.", Modality.MIXED, ["animal"]),
            ("The voices tell me to go outside and meet them.", Modality.AUDITORY, []),
            ("I feel strange today.", Modality.OTHER, []),
            ("My phone camera flashes at me.", Modality.VISUAL, ["camera", "phone"]),
        ]
        for text, modality, entities in cases:
            with self.subTest(text=text):
                claim = parse_claim(text)
                self.assertIs(claim.modality, modality)
                self.assertEqual(sorted(claim.entities), sorted(entities))

    def test_man_inside_woman_is_not_a_second_entity(self):
        claim = parse_claim("I saw a woman in the hallway.")
        self.assertEqual(claim.entities, ["person"])

    def test_command_and_check_intent(self):
        self.assertTrue(parse_claim("The voices tell me to hurt them.").is_command)
        self.assertFalse(parse_claim("I hear a voice whispering.").is_command)
        self.assertTrue(parse_claim("Is this real?").asks_for_check)
        self.assertFalse(parse_claim("The light is flickering.").asks_for_check)


if __name__ == "__main__":
    unittest.main()
