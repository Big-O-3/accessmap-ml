"""
test_app.py — tests for the Flask endpoints in app.py.

app.py imports detector, which calls Hugging Face's Inference API in
production. To keep these HTTP-layer tests fast and offline, we monkeypatch
app_module.run_analyze with a canned response — that's the single seam
between the route and the model.

Run:
    source venv/bin/activate
    pytest test_app.py
"""

import io
import unittest

import app as app_module
from detector import ModelUnavailableError


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
        self._original = app_module.run_analyze

    def tearDown(self):
        app_module.run_analyze = self._original

    def _stub_detect(self, detections, is_venue=True, framing_hint=None):
        """Point app.run_analyze at a canned result, ignoring the saved file."""
        app_module.run_analyze = lambda _path: {
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

    def test_model_unavailable_returns_503(self):
        def _raise(_path):
            raise ModelUnavailableError("HF is warming up")
        app_module.run_analyze = _raise
        data = {"image": (io.BytesIO(b"bytes"), "photo.jpg")}
        response = self.client.post(
            "/analyze", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 503)
        self.assertIn("warming up", response.get_json()["error"])


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
            {"cocoLabel": "door"},
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
