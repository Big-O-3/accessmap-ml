"""
detector.py — the accessibility feature detector.

We use Grounding DINO, an open-vocabulary object detector, and hand it a plain-
English prompt listing what we care about (ramps, doors, stairs, seating…). It
returns boxes matched to the prompt phrases.

In production we call it through Hugging Face's Serverless Inference API rather
than downloading the model — this keeps the deploy image ~50 MB instead of
~2 GB and works on any free-tier host. Requires the HF_API_TOKEN env var.

Kept separate from the web server (app.py) so you can test the model on its own:

    HF_API_TOKEN=hf_xxx python -c "from detector import analyze; \
        import json; print(json.dumps(analyze('samples/test.jpg'), indent=2))"

Output shape (the frontend + Node backend depend on it):

    {
      "cocoLabel": "a handrail",            # the phrase Grounding DINO matched
      "accessibilityFeature": "stairs_present",
      "confidence": 0.51,                   # 0.0 - 1.0
      "highConfidence": false,
      "boundingBox": {"x": 120, "y": 180, "width": 160, "height": 260}
    }

The "cocoLabel" field name is a legacy of the DB/API contract — Grounding DINO
labels aren't COCO classes, but the field is baked into the Prisma schema and
frontend, so it's kept as-is.
"""

import os

import requests
from PIL import Image

from features import PROMPT_TEXT, VENUE_KEYWORDS, to_feature

# Grounding DINO "tiny" — the fastest checkpoint, already detects the features
# we care about well. Called via HF's Serverless Inference API.
_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_HF_URL = f"https://api-inference.huggingface.co/models/{_MODEL_ID}"

# What we record as the model version on each analysis (surfaced to the DB).
MODEL_VERSION = "grounding-dino-tiny"

# Detection thresholds — matched to what post_process_grounded_object_detection
# uses locally so scores stay comparable across the API + local paths.
MIN_CONFIDENCE = 0.30
HIGH_CONFIDENCE = 0.5

# Cold-start / retry knobs. HF's serverless models sleep after ~15 min of
# inactivity; the first call after that returns 503 with a warm-up message. We
# retry a few times so the frontend just sees a slow first request, not a fail.
_TIMEOUT = 60           # seconds per HTTP call
_MAX_RETRIES = 6        # ~90 seconds total worst case
_RETRY_BACKOFF = 15     # seconds between retries when the model is warming


class ModelUnavailableError(RuntimeError):
    """Raised when the HF Inference API can't serve a detection."""


def _hf_headers():
    token = os.environ.get("HF_API_TOKEN")
    if not token:
        raise ModelUnavailableError(
            "HF_API_TOKEN is not set — get a Read token from "
            "https://huggingface.co/settings/tokens and add it to your env."
        )
    return {"Authorization": f"Bearer {token}"}


def _post_image(image_bytes):
    """POST raw image bytes to the HF Inference API, retrying cold-starts.

    HF returns 200 with the detections list, 503 while the model warms
    (retryable), or any other 4xx/5xx we surface as an error.
    """
    headers = _hf_headers()
    # Grounding DINO takes both the image and the text prompt. HF's API accepts
    # the prompt as an "inputs" text field on the same multipart body.
    for attempt in range(_MAX_RETRIES):
        response = requests.post(
            _HF_URL,
            headers=headers,
            data=image_bytes,
            params={"text": PROMPT_TEXT},
            timeout=_TIMEOUT,
        )
        if response.status_code == 200:
            return response.json()
        # 503 while the model spins up. HF sometimes signals this via a JSON
        # body with an "estimated_time" field.
        if response.status_code == 503:
            if attempt < _MAX_RETRIES - 1:
                import time
                time.sleep(_RETRY_BACKOFF)
                continue
        raise ModelUnavailableError(
            f"HF Inference API returned {response.status_code}: {response.text[:200]}"
        )
    raise ModelUnavailableError("HF Inference API is still warming up. Try again.")


def _shape_detections(raw_boxes):
    """Turn raw HF detections into the frontend detection list.

    Each raw entry from the HF API looks like:
        { "score": 0.51, "label": "a handrail",
          "box": {"xmin": 120, "ymin": 180, "xmax": 280, "ymax": 440} }

    Skips detections whose label has no mapped accessibility feature — those
    include the venue-gate keywords (building, storefront, sign, window), which
    are read separately by is_venue().
    """
    detections = []
    for entry in raw_boxes:
        label = entry.get("label", "")
        score = float(entry.get("score", 0.0))
        if score < MIN_CONFIDENCE:
            continue

        feature = to_feature(label)
        if feature is None:
            continue

        box = entry.get("box", {})
        x1 = box.get("xmin", 0)
        y1 = box.get("ymin", 0)
        x2 = box.get("xmax", 0)
        y2 = box.get("ymax", 0)
        bounding_box = {
            "x": round(x1),
            "y": round(y1),
            "width": round(x2 - x1),
            "height": round(y2 - y1),
        }

        confidence = round(score, 2)
        detections.append(
            {
                "cocoLabel": label,
                "accessibilityFeature": feature,
                "confidence": confidence,
                "highConfidence": confidence >= HIGH_CONFIDENCE,
                "boundingBox": bounding_box,
            }
        )

    return detections


def _run_model(image_path):
    """Send an image to the HF Inference API and return (raw_boxes, image_size).

    image_size is (width, height) in pixels — used by framing logic.
    """
    # Read the image size locally so framing_hint has real dimensions to work
    # with. The HF API returns boxes in the input image's coordinate space.
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_size = image.size

    with open(image_path, "rb") as fh:
        image_bytes = fh.read()

    raw = _post_image(image_bytes)
    return raw, image_size


def detect(image_path):
    """Run the detector on an image and return a list of detection dicts.

    Kept for backward compatibility (test_detect.py, direct callers). New
    callers should use analyze() to also get the venue gate + framing hint.
    """
    raw, _size = _run_model(image_path)
    return _shape_detections(raw)


def is_venue(raw_boxes):
    """Decide whether the photo looks like a venue at all.

    A shot passes the gate if ANY raw detection (feature-mapped or not) matches
    a venue keyword above MIN_CONFIDENCE. This uses the raw response rather
    than the shaped detections so unmapped labels like "building" or
    "storefront" still count.
    """
    for entry in raw_boxes:
        score = float(entry.get("score", 0.0))
        if score < MIN_CONFIDENCE:
            continue
        text = entry.get("label", "").lower()
        for kw in VENUE_KEYWORDS:
            if kw in text:
                return True
    return False


def framing_hint(detections, image_size):
    """Suggest how to reframe if the entrance is poorly positioned.

    Uses whichever entrance-like detection is largest. Returns None when
    there's nothing to base a hint on (no entrance detected) or the framing
    is already fine.
    """
    width, height = image_size
    frame_area = max(1, width * height)

    entrance_boxes = [
        d["boundingBox"]
        for d in detections
        if d["accessibilityFeature"] == "entrance_detected"
    ]
    if not entrance_boxes:
        return None

    box = max(entrance_boxes, key=lambda b: b["width"] * b["height"])
    area_ratio = (box["width"] * box["height"]) / frame_area

    # Cropped-off entrance: any edge of the box coincides with the image edge.
    touches_left = box["x"] <= 0
    touches_top = box["y"] <= 0
    touches_right = box["x"] + box["width"] >= width
    touches_bottom = box["y"] + box["height"] >= height
    if touches_left or touches_top or touches_right or touches_bottom:
        return "Move back or recenter — the entrance is cut off at the edge of the frame."

    if area_ratio > 0.6:
        return "Step back — the entrance fills most of the frame."
    if area_ratio < 0.03:
        return "Step closer — the entrance is very small in the frame."
    return None


def analyze(image_path):
    """Run the model once and return detections plus framing metadata.

    Response shape:
      {
        "detections":  [...],       # same as detect()
        "isVenue":     bool,        # False -> the photo doesn't look like a venue
        "framingHint": str | None,  # short suggestion, or None if framing is fine
      }
    """
    raw, size = _run_model(image_path)
    detections = _shape_detections(raw)
    return {
        "detections": detections,
        "isVenue": is_venue(raw),
        "framingHint": framing_hint(detections, size),
    }
