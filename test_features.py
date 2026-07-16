"""
test_features.py — unit tests for the label -> feature mapping.

These are pure-logic tests: no model, no network, no files. Run them with:

    source venv/bin/activate
    pytest test_features.py

features.py is the "dictionary" that connects the model's detected labels to the
frontend feature keys. We use Grounding DINO, which returns free-form labels
(e.g. "a handrail", "stairs steps"), so to_feature() matches by keyword. The
most valuable things to guard are that known labels resolve and unknown ones
are skipped.
"""

import unittest

from features import KEYWORD_MAP, PROMPT_TEXT, to_feature

# The feature keys the frontend knows how to render (mirrors
# accessmap-frontend-/src/lib/features.js). A mapped value outside this set
# would show a raw key in the UI, so we assert against this allow-list.
KNOWN_FRONTEND_KEYS = {
    "entrance_detected",
    "restroom_available",
    "parking_area",
    "seating_available",
    "indoor_seating",
    "stairs_present",
}


class ToFeatureTests(unittest.TestCase):
    def test_specific_mappings(self):
        # Explicit spot-checks so the intent is documented.
        self.assertEqual(to_feature("door"), "entrance_detected")
        self.assertEqual(to_feature("wheelchair ramp"), "entrance_detected")
        self.assertEqual(to_feature("stairs"), "stairs_present")
        self.assertEqual(to_feature("chair"), "seating_available")
        self.assertEqual(to_feature("toilet"), "restroom_available")

    def test_matches_multiword_and_partial_labels(self):
        # Grounding DINO returns labels like these; keyword matching must still
        # resolve them.
        self.assertEqual(to_feature("a handrail"), "stairs_present")
        self.assertEqual(to_feature("stairs steps"), "stairs_present")
        self.assertEqual(to_feature("ramp ramp"), "entrance_detected")

    def test_matching_is_case_insensitive(self):
        # to_feature lowercases the label, so casing shouldn't matter.
        self.assertEqual(to_feature("DOOR"), "entrance_detected")
        self.assertEqual(to_feature("Chair"), "seating_available")

    def test_unknown_label_returns_none(self):
        # An unmapped label must return None so detect() can skip it.
        self.assertIsNone(to_feature("dog"))
        self.assertIsNone(to_feature(""))
        self.assertIsNone(to_feature("banana"))


class TableConsistencyTests(unittest.TestCase):
    def test_all_features_are_known_frontend_keys(self):
        # Every mapped feature must be something the frontend can render.
        for keyword, feature in KEYWORD_MAP:
            with self.subTest(keyword=keyword):
                self.assertIn(feature, KNOWN_FRONTEND_KEYS)

    def test_every_keyword_appears_in_the_prompt(self):
        # A keyword we map but never ask the model to look for is dead weight.
        # (Compound labels like "staircase" contain "stairs"; we check the
        # keyword's presence as a substring of the prompt text.)
        prompt = PROMPT_TEXT.lower()
        for keyword, _feature in KEYWORD_MAP:
            with self.subTest(keyword=keyword):
                self.assertIn(keyword, prompt)

    def test_stairs_and_handrail_flag_barrier(self):
        # Stairs/steps/handrails should map to the barrier feature, since they
        # signal an accessibility obstacle.
        self.assertEqual(to_feature("steps"), "stairs_present")
        self.assertEqual(to_feature("handrail"), "stairs_present")

    def test_prompt_text_is_period_separated(self):
        # Grounding DINO expects concepts separated by periods.
        self.assertIn(".", PROMPT_TEXT)


if __name__ == "__main__":
    unittest.main()
