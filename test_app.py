"""
test_app.py — tests for the Flask endpoints in app.py.

app.py imports detector, which loads the real model at import time. As in
test_detector.py, we install a fake `ultralytics` first so no weights are
loaded. We then monkeypatch detector.detect via app's reference to it, so the
HTTP layer is tested independently of the model: we control exactly what
"detections" the endpoint sees and assert on status codes and JSON.

Run:
    source venv/bin/activate
    python -m unittest test_app
"""

import io
import sys
import types
import unittest


# Stub ultralytics before importing app -> detector (same reason as
# test_detector.py: avoid loading the 26 MB weights just to test HTTP).
_fake_ultralytics = types.ModuleType("ultralytics")


class _StubYOLOWorld:
    def __init__(self, *_args, **_kwargs):
        pass

    def set_classes(self, _classes):
        pass

    def predict(self, *_args, **_kwargs):
        return []


_fake_ultralytics.YOLOWorld = _StubYOLOWorld
sys.modules["ultralytics"] = _fake_ultralytics

import app as app_module  # noqa: E402


class HealthEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def test_health_returns_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})


class AnalyzeEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()
        # Remember the real detect so we can restore it after each test.
        self._original_detect = app_module.detect

    def tearDown(self):
        app_module.detect = self._original_detect

    def _stub_detect(self, detections):
        """Point app.detect at a canned result, ignoring the saved file."""
        app_module.detect = lambda _path: detections

    def test_missing_image_field_returns_400(self):
        response = self.client.post("/analyze", data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_empty_filename_returns_400(self):
        # An "image" part with a blank filename should be rejected.
        data = {"image": (io.BytesIO(b""), "")}
        response = self.client.post(
            "/analyze", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_valid_upload_returns_detections_and_alt_text(self):
        self._stub_detect(
            [
                {
                    "cocoLabel": "door",
                    "accessibilityFeature": "entrance_detected",
                    "confidence": 0.94,
                    "highConfidence": True,
                    "boundingBox": {"x": 1, "y": 2, "width": 3, "height": 4},
                }
            ]
        )
        data = {"image": (io.BytesIO(b"fake-jpeg-bytes"), "photo.jpg")}
        response = self.client.post(
            "/analyze", data=data, content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(len(body["detections"]), 1)
        self.assertEqual(body["detections"][0]["cocoLabel"], "door")
        self.assertEqual(body["altTextSuggestion"], "Detected: door.")


class BuildAltTextTests(unittest.TestCase):
    def test_empty_detections(self):
        self.assertEqual(
            app_module.build_alt_text([]),
            "No accessibility features detected.",
        )

    def test_lists_unique_labels_in_first_seen_order(self):
        detections = [
            {"cocoLabel": "door"},
            {"cocoLabel": "chair"},
            {"cocoLabel": "door"},  # duplicate should not repeat
        ]
        self.assertEqual(
            app_module.build_alt_text(detections),
            "Detected: door, chair.",
        )

    def test_single_detection(self):
        self.assertEqual(
            app_module.build_alt_text([{"cocoLabel": "toilet"}]),
            "Detected: toilet.",
        )


if __name__ == "__main__":
    unittest.main()
