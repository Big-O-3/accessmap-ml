"""
detector.py — the accessibility feature detector.

We use Grounding DINO, an open-vocabulary object detector. It takes a plain-
English prompt listing what we care about (ramps, doors, stairs, seating…)
and returns boxes matched to those phrases.

Currently running the model LOCALLY via `transformers` + `torch` (weights are
downloaded on first use, ~700 MB, cached to ~/.cache/huggingface). This is
what runs both in local dev and on Render (Starter plan, 2 GB RAM).

The HF Inference API code path is preserved further down (commented out).
As of 2026-07 HF's serverless provider no longer hosts Grounding DINO, so
that path is currently non-functional — kept for reference in case they add
it back or for switching to a paid provider like Replicate.

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

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from features import PROMPT_TEXT, VENUE_KEYWORDS, to_feature

# The Grounding DINO checkpoint. "tiny" is the fastest; it already detects our
# accessibility features well. Downloads (~700MB) and caches on first run.
_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# What we record as the model version on each analysis (surfaced to the DB).
MODEL_VERSION = "grounding-dino-tiny"

# Detection thresholds — matched so scores stay comparable across paths.
MIN_CONFIDENCE = 0.30
# Text-match threshold — how strongly a box must match the prompt phrase. 0.30
# reduces duplicate/garbled labels like "ramp ramp".
TEXT_THRESHOLD = 0.30

# At or above this confidence, a detection is treated as "high confidence" and
# the frontend pre-checks it for the contributor.
HIGH_CONFIDENCE = 0.5


class ModelUnavailableError(RuntimeError):
    """Raised when the detector can't serve a request (kept for app.py compat)."""


# --- LOCAL MODEL PATH -------------------------------------------------------
# Loads Grounding DINO into memory once (slow first run, fast every call after).

_processor = AutoProcessor.from_pretrained(_MODEL_ID)
_model = AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_ID)


def _run_model(image_path):
    """Run Grounding DINO on an image and return (raw_boxes, image_size).

    raw_boxes uses the same shape as the HF Inference API response so the
    downstream _shape_detections / is_venue code is path-agnostic:
        [{"label": str, "score": float,
          "box": {"xmin": ..., "ymin": ..., "xmax": ..., "ymax": ...}}, ...]

    image_size is (width, height) in pixels — used by framing logic.
    """
    image = Image.open(image_path).convert("RGB")

    inputs = _processor(images=image, text=PROMPT_TEXT, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)

    results = _processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=MIN_CONFIDENCE,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]

    raw_boxes = []
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        x1, y1, x2, y2 = box.tolist()
        raw_boxes.append(
            {
                "label": label,
                "score": float(score),
                "box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
            }
        )
    return raw_boxes, image.size


# --- HF INFERENCE API PATH (currently unsupported; commented for reference) -
# As of 2026-07 HF's free hf-inference provider no longer serves
# Grounding DINO. Kept here in case a paid provider (Replicate, Together) is
# wired up later via HF's router.
#
# import requests
#
# _HF_URL = f"https://router.huggingface.co/hf-inference/models/{_MODEL_ID}"
# _TIMEOUT = 60
# _MAX_RETRIES = 6
# _RETRY_BACKOFF = 15
#
#
# def _hf_headers():
#     token = os.environ.get("HF_API_TOKEN")
#     if not token:
#         raise ModelUnavailableError(
#             "HF_API_TOKEN is not set — get a Read token from "
#             "https://huggingface.co/settings/tokens and add it to your env."
#         )
#     return {"Authorization": f"Bearer {token.strip()}"}
#
#
# def _post_image(image_bytes):
#     headers = _hf_headers()
#     for attempt in range(_MAX_RETRIES):
#         response = requests.post(
#             _HF_URL, headers=headers, data=image_bytes,
#             params={"text": PROMPT_TEXT}, timeout=_TIMEOUT,
#         )
#         if response.status_code == 200:
#             return response.json()
#         if response.status_code == 503 and attempt < _MAX_RETRIES - 1:
#             import time
#             time.sleep(_RETRY_BACKOFF)
#             continue
#         raise ModelUnavailableError(
#             f"HF Inference API returned {response.status_code}: {response.text[:200]}"
#         )
#     raise ModelUnavailableError("HF Inference API is still warming up. Try again.")
#
#
# def _run_model(image_path):
#     with Image.open(image_path) as image:
#         image = image.convert("RGB")
#         image_size = image.size
#     with open(image_path, "rb") as fh:
#         image_bytes = fh.read()
#     raw = _post_image(image_bytes)
#     return raw, image_size


# --- Path-agnostic helpers --------------------------------------------------

def _shape_detections(raw_boxes):
    """Turn raw model output into the frontend detection list.

    Each raw entry: {"label", "score", "box": {"xmin","ymin","xmax","ymax"}}.

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


def detect(image_path):
    """Run the detector on an image and return a list of detection dicts.

    Kept for backward compatibility (test_detect.py, direct callers). New
    callers should use analyze() to also get the venue gate + framing hint.
    """
    raw, _size = _run_model(image_path)
    return _shape_detections(raw)


def is_venue(raw_boxes):
    """Decide whether the photo looks like a venue at all.

    A shot passes the gate if ANY raw detection (feature-mapped or not)
    matches a venue keyword above MIN_CONFIDENCE.
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
