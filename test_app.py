"""
test_app.py — tests for the Flask endpoints in app.py.

app.py imports detector, which loads the real Grounding DINO model at import
time. To keep these HTTP-layer tests fast, we install fake `torch`,
`transformers`, and `PIL` modules before importing app (same approach as
test_detector.py) so no weights are downloaded. We then monkeypatch
app.detect to control exactly what "detections" the endpoint sees, and assert
on status codes and JSON.

Run:
    source venv/bin/activate
    pytest test_app.py
"""

import io
import sys
import types
import unittest


def _install_fakes():
    """Stub the heavy ML deps so importing app -> detector is instant."""
    fake_torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    fake_torch.no_grad = lambda: _NoGrad()
    sys.modules["torch"] = fake_torch

    fake_tf = types.ModuleType("transformers")

    class _Stub:
        def __call__(self, *a, **k):
            return {"input_ids": [[0]]}

        def post_process_grounded_object_detection(self, *a, **k):
            return [{"boxes": [], "labels": [], "scores": []}]

    fake_tf.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **k: _Stub())
    fake_tf.AutoModelForZeroShotObjectDetection = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _Stub()
    )
    sys.modules["transformers"] = fake_tf

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
        self._original_analyze = app_module.analyze

    def tearDown(self):
        app_module.analyze = self._original_analyze

    def _stub_detect(self, detections, is_venue=True, framing_hint=None):
        """Point app.analyze at a canned result, ignoring the saved file."""
        app_module.analyze = lambda _path: {
            "detections": detections,
            "isVenue": is_venue,
            "framingHint": framing_hint,
        }

    def test_missing_image_field_returns_400(self):
        response = self.client.post("/analyze", data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_empty_filename_returns_400(self):
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

    def test_valid_upload_with_no_detections(self):
        self._stub_detect([])
        data = {"image": (io.BytesIO(b"bytes"), "empty.jpg")}
        response = self.client.post(
            "/analyze", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["detections"], [])
        self.assertEqual(body["altTextSuggestion"], "No accessibility features detected.")

    def test_response_surfaces_is_venue_and_framing_hint(self):
        self._stub_detect(
            [
                {
                    "cocoLabel": "door",
                    "accessibilityFeature": "entrance_detected",
                    "confidence": 0.94,
                    "highConfidence": True,
                    "boundingBox": {"x": 1, "y": 2, "width": 3, "height": 4},
                }
            ],
            is_venue=True,
            framing_hint="Step closer — the entrance is very small in the frame.",
        )
        data = {"image": (io.BytesIO(b"bytes"), "photo.jpg")}
        response = self.client.post(
            "/analyze", data=data, content_type="multipart/form-data"
        )
        body = response.get_json()
        self.assertTrue(body["isVenue"])
        self.assertIn("Step closer", body["framingHint"])

    def test_response_when_photo_is_not_a_venue(self):
        self._stub_detect([], is_venue=False, framing_hint=None)
        data = {"image": (io.BytesIO(b"bytes"), "selfie.jpg")}
        response = self.client.post(
            "/analyze", data=data, content_type="multipart/form-data"
        )
        body = response.get_json()
        self.assertFalse(body["isVenue"])
        self.assertIsNone(body["framingHint"])


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
