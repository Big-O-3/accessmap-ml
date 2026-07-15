"""
test_features.py — unit tests for the prompt -> feature mapping.

These are pure-logic tests: no model, no network, no files. Run them with:

    source venv/bin/activate
    python -m unittest test_features

features.py is the "dictionary" that connects the model's prompts to the
frontend feature keys, so the most valuable thing to guard is that the two
tables (PROMPTS and FEATURE_MAP) stay in sync — the module's own docstring
warns that a mismatch silently shows a raw key in the UI.
"""

import unittest

from features import FEATURE_MAP, PROMPTS, to_feature

# The feature keys the frontend knows how to render (from the comment in
# features.py, which mirrors accessmap-frontend-/src/lib/features.js). If a
# mapped value isn't in here, the frontend would show the raw key instead of a
# friendly label — so we assert against this allow-list.
KNOWN_FRONTEND_KEYS = {
    "entrance_detected",
    "restroom_available",
    "parking_area",
    "seating_available",
    "indoor_seating",
    "stairs_present",
}


class ToFeatureTests(unittest.TestCase):
    def test_maps_each_known_prompt_to_its_feature(self):
        # Every prompt in FEATURE_MAP should resolve to its mapped value.
        for prompt, expected in FEATURE_MAP.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(to_feature(prompt), expected)

    def test_specific_mappings(self):
        # A few explicit spot-checks so the intent is documented, not just
        # derived from the table itself.
        self.assertEqual(to_feature("door"), "entrance_detected")
        self.assertEqual(to_feature("wheelchair ramp"), "entrance_detected")
        self.assertEqual(to_feature("stairs"), "stairs_present")
        self.assertEqual(to_feature("chair"), "seating_available")
        self.assertEqual(to_feature("toilet"), "restroom_available")

    def test_unknown_prompt_returns_none(self):
        # An unmapped label must return None so detect() can skip it.
        self.assertIsNone(to_feature("dog"))
        self.assertIsNone(to_feature(""))
        self.assertIsNone(to_feature("DOOR"))  # lookup is case-sensitive


class TableConsistencyTests(unittest.TestCase):
    def test_every_prompt_is_mapped(self):
        # The docstring in features.py says each PROMPT must appear in
        # FEATURE_MAP. A missing entry means the model looks for something the
        # frontend can never display.
        for prompt in PROMPTS:
            with self.subTest(prompt=prompt):
                self.assertIn(prompt, FEATURE_MAP)
                self.assertIsNotNone(to_feature(prompt))

    def test_no_orphan_map_entries(self):
        # The reverse guard: every mapped prompt should be something we
        # actually ask the model to look for. An orphan entry is dead weight.
        for prompt in FEATURE_MAP:
            with self.subTest(prompt=prompt):
                self.assertIn(prompt, PROMPTS)

    def test_all_features_are_known_frontend_keys(self):
        for prompt, feature in FEATURE_MAP.items():
            with self.subTest(prompt=prompt):
                self.assertIn(feature, KNOWN_FRONTEND_KEYS)

    def test_no_duplicate_prompts(self):
        self.assertEqual(len(PROMPTS), len(set(PROMPTS)))


if __name__ == "__main__":
    unittest.main()
