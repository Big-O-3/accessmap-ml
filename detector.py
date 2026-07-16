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

from features import PROMPT_TEXT, to_feature

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


def detect(image_path):
    """Run Grounding DINO on an image and return a list of detection dicts.

    Each dict is shaped for the frontend. Detections whose label has no mapped
    accessibility feature are skipped.
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
