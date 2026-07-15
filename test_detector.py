"""
test_detector.py — unit tests for detect() in detector.py.

detector.py loads the real YOLO-World model at import time, which is slow and
needs the 26 MB weights. We don't want any of that just to test the box ->
dict shaping logic, so before importing detector we install a fake
`ultralytics` module into sys.modules. The fake YOLOWorld records set_classes()
calls and returns whatever boxes each test hands it — so we can drive detect()
with precise, synthetic detections and assert on the exact output shape the
frontend depends on.

Run:
    source venv/bin/activate
    python -m unittest test_detector
"""

import sys
import types
import unittest


# --- Fakes standing in for ultralytics / torch box objects -----------------

class FakeScalar:
    """Mimics a 0-d tensor: int()/float() unwrap it, like box.cls / box.conf."""

    def __init__(self, value):
        self._value = value

    def __int__(self):
        return int(self._value)

    def __float__(self):
        return float(self._value)


class FakeXYXY:
    """Mimics box.xyxy — indexing [0] gives an object with .tolist()."""

    def __init__(self, coords):
        self._coords = coords

    def __getitem__(self, idx):
        assert idx == 0
        return types.SimpleNamespace(tolist=lambda: list(self._coords))


class FakeBox:
    def __init__(self, cls_index, conf, xyxy):
        self.cls = FakeScalar(cls_index)
        self.conf = FakeScalar(conf)
        self.xyxy = FakeXYXY(xyxy)


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class FakeYOLOWorld:
    """Records set_classes() and returns boxes queued by the test."""

    # Class-level queue so a test can set what the next predict() returns
    # before detect() (which owns the real instance) calls it.
    next_boxes = []

    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.classes = []

    def set_classes(self, classes):
        self.classes = list(classes)

    def predict(self, image_path, conf=0.25, verbose=True):
        # names maps class index -> the prompt set via set_classes.
        names = {i: c for i, c in enumerate(self.classes)}
        return [FakeResult(list(FakeYOLOWorld.next_boxes), names)]


# Install the fake BEFORE importing detector, so its module-level
# YOLOWorld("yolov8s-world.pt") uses the stub instead of loading real weights.
_fake_ultralytics = types.ModuleType("ultralytics")
_fake_ultralytics.YOLOWorld = FakeYOLOWorld
sys.modules["ultralytics"] = _fake_ultralytics

import detector  # noqa: E402  (must come after the stub is installed)
from features import PROMPTS  # noqa: E402


def box_for(prompt, conf, xyxy):
    """Build a FakeBox whose class index matches `prompt`'s slot in PROMPTS."""
    return FakeBox(PROMPTS.index(prompt), conf, xyxy)


class DetectTests(unittest.TestCase):
    def tearDown(self):
        FakeYOLOWorld.next_boxes = []

    def test_model_configured_with_prompts_at_import(self):
        # detector sets the open-vocabulary classes to our PROMPTS on load.
        self.assertEqual(detector._model.classes, PROMPTS)
        self.assertEqual(detector._model.weights_path, "yolov8s-world.pt")

    def test_no_boxes_returns_empty_list(self):
        FakeYOLOWorld.next_boxes = []
        self.assertEqual(detector.detect("anything.jpg"), [])

    def test_shapes_detection_for_frontend(self):
        # A "door" at pixel box [120, 180, 280, 440] -> x/y/width/height.
        FakeYOLOWorld.next_boxes = [box_for("door", 0.94, [120, 180, 280, 440])]
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
        FakeYOLOWorld.next_boxes = [box_for("chair", 0.21678, [0, 0, 10, 10])]
        result = detector.detect("img.jpg")
        self.assertEqual(result[0]["confidence"], 0.22)

    def test_high_confidence_flag_uses_threshold(self):
        # Exactly at the threshold counts as high confidence (>=).
        FakeYOLOWorld.next_boxes = [
            box_for("door", detector.HIGH_CONFIDENCE, [0, 0, 5, 5])
        ]
        self.assertTrue(detector.detect("img.jpg")[0]["highConfidence"])

        # Just below is not.
        FakeYOLOWorld.next_boxes = [
            box_for("door", detector.HIGH_CONFIDENCE - 0.01, [0, 0, 5, 5])
        ]
        self.assertFalse(detector.detect("img.jpg")[0]["highConfidence"])

    def test_bounding_box_coordinates_are_rounded(self):
        FakeYOLOWorld.next_boxes = [
            box_for("chair", 0.5, [10.4, 20.6, 50.4, 70.6])
        ]
        bbox = detector.detect("img.jpg")[0]["boundingBox"]
        self.assertEqual(bbox, {"x": 10, "y": 21, "width": 40, "height": 50})

    def test_unmapped_prompt_is_skipped(self):
        # Inject a class the FEATURE_MAP has no entry for by extending the
        # model's classes with an extra label at a known index.
        detector._model.classes = PROMPTS + ["unmapped thing"]
        try:
            extra_index = len(PROMPTS)  # points at "unmapped thing"
            FakeYOLOWorld.next_boxes = [
                FakeBox(extra_index, 0.99, [0, 0, 5, 5]),
                box_for("door", 0.90, [1, 1, 2, 2]),
            ]
            result = detector.detect("img.jpg")
            # Only the mapped "door" survives.
            self.assertEqual([d["cocoLabel"] for d in result], ["door"])
        finally:
            detector._model.classes = PROMPTS  # restore for other tests

    def test_multiple_detections_all_returned(self):
        FakeYOLOWorld.next_boxes = [
            box_for("chair", 0.30, [0, 0, 10, 10]),
            box_for("toilet", 0.40, [5, 5, 15, 15]),
        ]
        result = detector.detect("img.jpg")
        self.assertEqual(
            {d["accessibilityFeature"] for d in result},
            {"seating_available", "restroom_available"},
        )

    def test_predict_called_with_min_confidence_floor(self):
        # Capture the conf passed into predict to lock in the low floor.
        captured = {}
        original = detector._model.predict

        def spy(image_path, conf=0.25, verbose=True):
            captured["conf"] = conf
            return original(image_path, conf=conf, verbose=verbose)

        detector._model.predict = spy
        try:
            FakeYOLOWorld.next_boxes = []
            detector.detect("img.jpg")
            self.assertEqual(captured["conf"], detector.MIN_CONFIDENCE)
        finally:
            detector._model.predict = original


if __name__ == "__main__":
    unittest.main()
