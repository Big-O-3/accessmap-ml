"""
test_detector.py — unit tests for detect() and analyze() in detector.py.

The real detector POSTs an image to Hugging Face's Inference API. We don't
want to hit the network in tests, so we monkeypatch detector._run_model to
return canned raw boxes + a canned image size. That's the single seam between
the module and everything network/filesystem related.

Run:
    source venv/bin/activate
    pytest test_detector.py
"""

import unittest

import detector


# Fake 800x600 frame so framing math has real dimensions to work with.
_IMAGE_SIZE = (800, 600)

# Queue what detector._run_model will return; tests set this per case.
_NEXT_RAW = []


def _fake_run_model(_image_path):
    return _NEXT_RAW, _IMAGE_SIZE


detector._run_model = _fake_run_model


def set_boxes(entries):
    """Queue the raw boxes the fake HF response will return.

    Each entry: (label, score, x1, y1, x2, y2).
    """
    global _NEXT_RAW
    _NEXT_RAW = [
        {
            "label": label,
            "score": score,
            "box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
        }
        for (label, score, x1, y1, x2, y2) in entries
    ]


class DetectTests(unittest.TestCase):
    def tearDown(self):
        set_boxes([])

    def test_no_boxes_returns_empty_list(self):
        set_boxes([])
        self.assertEqual(detector.detect("anything.jpg"), [])

    def test_shapes_detection_for_frontend(self):
        set_boxes([("door", 0.94, 120, 180, 280, 440)])
        result = detector.detect("img.jpg")

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0],
            {
                "cocoLabel": "door",
                "accessibilityFeature": "entrance_detected",
                "confidence": 0.94,
                "highConfidence": True,
                "boundingBox": {"x": 120, "y": 180, "width": 160, "height": 260},
            },
        )

    def test_confidence_is_rounded_to_two_decimals(self):
        # Must clear MIN_CONFIDENCE (0.30) — the API path filters low scores.
        set_boxes([("chair", 0.41678, 0, 0, 10, 10)])
        result = detector.detect("img.jpg")
        self.assertEqual(result[0]["confidence"], 0.42)

    def test_high_confidence_flag_uses_threshold(self):
        set_boxes([("door", detector.HIGH_CONFIDENCE, 0, 0, 5, 5)])
        self.assertTrue(detector.detect("img.jpg")[0]["highConfidence"])

        set_boxes([("door", detector.HIGH_CONFIDENCE - 0.01, 0, 0, 5, 5)])
        self.assertFalse(detector.detect("img.jpg")[0]["highConfidence"])

    def test_bounding_box_coordinates_are_rounded(self):
        set_boxes([("chair", 0.5, 10.4, 20.6, 50.4, 70.6)])
        bbox = detector.detect("img.jpg")[0]["boundingBox"]
        self.assertEqual(bbox, {"x": 10, "y": 21, "width": 40, "height": 50})

    def test_multiword_label_maps_to_feature(self):
        set_boxes([("a handrail", 0.6, 0, 0, 5, 5)])
        result = detector.detect("img.jpg")
        self.assertEqual(result[0]["accessibilityFeature"], "stairs_present")

    def test_unmapped_label_is_skipped(self):
        set_boxes(
            [
                ("unmapped thing", 0.99, 0, 0, 5, 5),
                ("door", 0.90, 1, 1, 2, 2),
            ]
        )
        result = detector.detect("img.jpg")
        self.assertEqual([d["cocoLabel"] for d in result], ["door"])

    def test_low_confidence_detection_is_dropped(self):
        set_boxes([("door", detector.MIN_CONFIDENCE - 0.01, 0, 0, 5, 5)])
        self.assertEqual(detector.detect("img.jpg"), [])

    def test_multiple_detections_all_returned(self):
        set_boxes(
            [
                ("chair", 0.40, 0, 0, 10, 10),
                ("toilet", 0.50, 5, 5, 15, 15),
            ]
        )
        result = detector.detect("img.jpg")
        self.assertEqual(
            {d["accessibilityFeature"] for d in result},
            {"seating_available", "restroom_available"},
        )


class AnalyzeTests(unittest.TestCase):
    def tearDown(self):
        set_boxes([])

    def test_returns_expected_shape(self):
        set_boxes([])
        result = detector.analyze("img.jpg")
        self.assertEqual(set(result.keys()), {"detections", "isVenue", "framingHint"})

    def test_is_venue_true_when_storefront_detected(self):
        set_boxes([("a storefront", 0.55, 0, 0, 5, 5)])
        result = detector.analyze("img.jpg")
        self.assertEqual(result["detections"], [])
        self.assertTrue(result["isVenue"])

    def test_is_venue_true_when_door_detected(self):
        set_boxes([("door", 0.7, 10, 10, 40, 40)])
        self.assertTrue(detector.analyze("img.jpg")["isVenue"])

    def test_is_venue_false_when_nothing_venue_like(self):
        set_boxes([("chair", 0.6, 0, 0, 5, 5)])
        self.assertFalse(detector.analyze("img.jpg")["isVenue"])

    def test_is_venue_ignores_low_confidence_labels(self):
        set_boxes([("building", detector.MIN_CONFIDENCE - 0.05, 0, 0, 5, 5)])
        self.assertFalse(detector.analyze("img.jpg")["isVenue"])

    def test_framing_hint_step_back_when_entrance_fills_frame(self):
        set_boxes([("door", 0.9, 100, 50, 700, 550)])
        hint = detector.analyze("img.jpg")["framingHint"]
        self.assertIsNotNone(hint)
        self.assertIn("Step back", hint)

    def test_framing_hint_step_closer_when_entrance_is_tiny(self):
        set_boxes([("door", 0.9, 100, 100, 140, 140)])
        hint = detector.analyze("img.jpg")["framingHint"]
        self.assertIsNotNone(hint)
        self.assertIn("Step closer", hint)

    def test_framing_hint_recenter_when_entrance_touches_edge(self):
        set_boxes([("door", 0.9, 0, 100, 200, 400)])
        hint = detector.analyze("img.jpg")["framingHint"]
        self.assertIsNotNone(hint)
        self.assertIn("cut off", hint)

    def test_framing_hint_none_when_well_framed(self):
        set_boxes([("door", 0.9, 300, 150, 500, 450)])
        self.assertIsNone(detector.analyze("img.jpg")["framingHint"])

    def test_framing_hint_none_without_entrance(self):
        set_boxes([("chair", 0.6, 0, 0, 10, 10)])
        self.assertIsNone(detector.analyze("img.jpg")["framingHint"])


if __name__ == "__main__":
    unittest.main()
