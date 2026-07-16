"""
test_detector.py — unit tests for detect() in detector.py.

detector.py loads the real Grounding DINO model (via transformers) at import
time, which is slow and downloads weights. We don't want that just to test the
box -> dict shaping logic, so before importing detector we install fake
`torch`, `transformers`, and `PIL.Image` pieces into sys.modules. The fakes let
each test hand detect() precise, synthetic detections and assert on the exact
output shape the frontend depends on.

Run:
    source venv/bin/activate
    pytest test_detector.py
"""

import sys
import types
import unittest


# --- Fakes standing in for transformers / torch / PIL ------------------------

class FakeTensor:
    """Mimics a tensor whose .tolist() returns the wrapped coords."""

    def __init__(self, value):
        self._value = value

    def tolist(self):
        return list(self._value)

    def __float__(self):
        return float(self._value)


# The result dict that post_process_grounded_object_detection returns. Tests set
# this before calling detect().
_NEXT_RESULT = {"boxes": [], "labels": [], "scores": []}


class FakeProcessor:
    def __call__(self, images=None, text=None, return_tensors=None):
        # detector indexes inputs["input_ids"]; any object with that key works.
        return {"input_ids": [[0]]}

    def post_process_grounded_object_detection(self, *args, **kwargs):
        return [_NEXT_RESULT]


class FakeModel:
    def __call__(self, **kwargs):
        return object()  # detector only passes this straight to post_process


def _install_fakes():
    # torch: detector uses torch.no_grad() as a context manager.
    fake_torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake_torch.no_grad = lambda: _NoGrad()
    sys.modules["torch"] = fake_torch

    # transformers: detector calls AutoProcessor / AutoModel .from_pretrained.
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoProcessor = types.SimpleNamespace(
        from_pretrained=lambda *_a, **_k: FakeProcessor()
    )
    fake_tf.AutoModelForZeroShotObjectDetection = types.SimpleNamespace(
        from_pretrained=lambda *_a, **_k: FakeModel()
    )
    sys.modules["transformers"] = fake_tf

    # PIL.Image.open(...).convert("RGB").size -> a (w, h) tuple.
    fake_pil = types.ModuleType("PIL")
    fake_image_mod = types.ModuleType("PIL.Image")

    class _Img:
        size = (800, 600)

        def convert(self, _mode):
            return self

    fake_image_mod.open = lambda _path: _Img()
    fake_pil.Image = fake_image_mod
    sys.modules["PIL"] = fake_pil
    sys.modules["PIL.Image"] = fake_image_mod


_install_fakes()

# Another test module (e.g. test_app) may have already imported `detector`
# against a different stub. Drop any cached copy so it re-imports against OUR
# fakes and binds to our fake processor/model.
sys.modules.pop("detector", None)

import detector  # noqa: E402  (must come after the fakes are installed)


def set_result(boxes, labels, scores):
    """Queue what the fake processor's post-process step will return."""
    _NEXT_RESULT["boxes"] = [FakeTensor(b) for b in boxes]
    _NEXT_RESULT["labels"] = labels
    _NEXT_RESULT["scores"] = [FakeTensor(s) for s in scores]


class DetectTests(unittest.TestCase):
    def tearDown(self):
        set_result([], [], [])

    def test_no_boxes_returns_empty_list(self):
        set_result([], [], [])
        self.assertEqual(detector.detect("anything.jpg"), [])

    def test_shapes_detection_for_frontend(self):
        # A "door" at pixel box [120, 180, 280, 440] -> x/y/width/height.
        set_result([[120, 180, 280, 440]], ["door"], [0.94])
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
        set_result([[0, 0, 10, 10]], ["chair"], [0.21678])
        result = detector.detect("img.jpg")
        self.assertEqual(result[0]["confidence"], 0.22)

    def test_high_confidence_flag_uses_threshold(self):
        # At/above the threshold counts as high confidence (>=).
        set_result([[0, 0, 5, 5]], ["door"], [detector.HIGH_CONFIDENCE])
        self.assertTrue(detector.detect("img.jpg")[0]["highConfidence"])

        # Just below is not.
        set_result([[0, 0, 5, 5]], ["door"], [detector.HIGH_CONFIDENCE - 0.01])
        self.assertFalse(detector.detect("img.jpg")[0]["highConfidence"])

    def test_bounding_box_coordinates_are_rounded(self):
        set_result([[10.4, 20.6, 50.4, 70.6]], ["chair"], [0.5])
        bbox = detector.detect("img.jpg")[0]["boundingBox"]
        self.assertEqual(bbox, {"x": 10, "y": 21, "width": 40, "height": 50})

    def test_multiword_label_maps_to_feature(self):
        # Grounding DINO often returns labels like "a handrail".
        set_result([[0, 0, 5, 5]], ["a handrail"], [0.6])
        result = detector.detect("img.jpg")
        self.assertEqual(result[0]["accessibilityFeature"], "stairs_present")

    def test_unmapped_label_is_skipped(self):
        set_result(
            [[0, 0, 5, 5], [1, 1, 2, 2]],
            ["unmapped thing", "door"],
            [0.99, 0.90],
        )
        result = detector.detect("img.jpg")
        # Only the mapped "door" survives.
        self.assertEqual([d["cocoLabel"] for d in result], ["door"])

    def test_multiple_detections_all_returned(self):
        set_result(
            [[0, 0, 10, 10], [5, 5, 15, 15]],
            ["chair", "toilet"],
            [0.40, 0.50],
        )
        result = detector.detect("img.jpg")
        self.assertEqual(
            {d["accessibilityFeature"] for d in result},
            {"seating_available", "restroom_available"},
        )


if __name__ == "__main__":
    unittest.main()
