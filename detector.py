"""
detector.py — the actual AI. Loads Grounding DINO once and runs it on a photo.

We use Grounding DINO (an open-vocabulary detector) instead of YOLO-World
because YOLO-World has a blind spot for architectural accessibility features —
it could not detect ramps, stairs, or doors at any confidence, while Grounding
DINO reliably surfaces them.

Kept separate from the web server (app.py) so you can test the model on its own:

    python -c "from detector import detect; import json; print(json.dumps(detect('samples/test.jpg'), indent=2))"

The output shape is unchanged from before (the frontend + Node backend depend
on it):

    {
      "cocoLabel": "a handrail",            # the phrase Grounding DINO matched
      "accessibilityFeature": "stairs_present",
      "confidence": 0.51,                   # 0.0 - 1.0
      "highConfidence": false,
      "boundingBox": {"x": 120, "y": 180, "width": 160, "height": 260}
    }
"""

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from features import PROMPT_TEXT, VENUE_KEYWORDS, to_feature

# The Grounding DINO checkpoint. "tiny" is the fastest; it already detects our
# accessibility features well. Downloads (~700MB) and caches on first run.
_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# What we record as the model version on each analysis (surfaced to the DB).
MODEL_VERSION = "grounding-dino-tiny"

# Load the processor + model ONCE at import time (loading is slow).
_processor = AutoProcessor.from_pretrained(_MODEL_ID)
_model = AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_ID)

# Minimum box confidence to return a detection. Tuned to 0.30: real features
# (ramps, handrails, stairs, seating) survive while borderline noise is cut.
MIN_CONFIDENCE = 0.30

# Text-match threshold — how strongly a box must match the prompt phrase. 0.30
# reduces duplicate/garbled labels like "ramp ramp".
TEXT_THRESHOLD = 0.30

# At or above this confidence, a detection is treated as "high confidence" and
# the frontend pre-checks it for the contributor.
HIGH_CONFIDENCE = 0.5


def _run_model(image_path):
    """Load an image, run Grounding DINO, return (results, image_size).

    image_size is (width, height) in pixels — used by framing logic.
    """
    image = Image.open(image_path).convert("RGB")

    inputs = _processor(images=image, text=PROMPT_TEXT, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**inputs)

    # post_process returns boxes in (x1, y1, x2, y2) pixel coords, plus the
    # matched label text and a score, for the original image size.
    results = _processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=MIN_CONFIDENCE,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],  # (height, width)
    )[0]

    return results, image.size


def _shape_detections(results):
    """Turn raw model results into the frontend detection list.

    Skips detections whose label has no mapped accessibility feature — those
    include the venue-gate keywords (building, storefront, sign, window), which
    are read separately by is_venue().
    """
    detections = []
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        feature = to_feature(label)
        if feature is None:
            continue

        x1, y1, x2, y2 = box.tolist()
        bounding_box = {
            "x": round(x1),
            "y": round(y1),
            "width": round(x2 - x1),
            "height": round(y2 - y1),
        }

        confidence = round(float(score), 2)
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
    """Run Grounding DINO on an image and return a list of detection dicts.

    Kept for backward compatibility (test_detect.py, direct callers). New
    callers should use analyze() to also get the venue gate + framing hint.
    """
    results, _size = _run_model(image_path)
    return _shape_detections(results)


def is_venue(results):
    """Decide whether the photo looks like a venue at all.

    A shot passes the gate if ANY raw detection (feature-mapped or not)
    matches a venue keyword above MIN_CONFIDENCE. This uses the raw results
    rather than the shaped detections list so unmapped labels like "building"
    or "storefront" still count.
    """
    for label, score in zip(results["labels"], results["scores"]):
        text = label.lower()
        if float(score) < MIN_CONFIDENCE:
            continue
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
    results, size = _run_model(image_path)
    detections = _shape_detections(results)
    return {
        "detections": detections,
        "isVenue": is_venue(results),
        "framingHint": framing_hint(detections, size),
    }
