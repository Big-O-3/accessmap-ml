"""
detector.py — the actual AI. Loads YOLO-World once and runs it on a photo.

Kept separate from the web server (app.py) so you can test the model on its own:

    python -c "from detector import detect; import json; print(json.dumps(detect('samples/test.jpg'), indent=2))"

The output is shaped to match exactly what the React frontend expects (see
accessmap-frontend-/src/components/DetectionImage.jsx):

    {
      "cocoLabel": "door",                 # the prompt that matched
      "accessibilityFeature": "entrance_detected",
      "confidence": 0.94,                  # 0.0 - 1.0
      "boundingBox": {"x": 120, "y": 180, "width": 160, "height": 260}
    }
"""

from ultralytics import YOLOWorld

from features import PROMPTS, to_feature

# Load the model ONCE when this file is first imported. Loading is slow, so we
# do NOT want to reload it on every request. The first run auto-downloads the
# weights file (yolov8s-world.pt, ~tens of MB) and caches it locally.
_model = YOLOWorld("yolov8s-world.pt")

# Tell the open-vocabulary model which phrases to look for.
_model.set_classes(PROMPTS)

# Minimum confidence for a detection to be returned at all. Open-vocabulary
# models like YOLO-World tend to score lower than fixed-class models, and
# ultralytics defaults to 0.25 (which drops most of our accessibility
# detections). We use a lower floor so borderline features still surface; the
# frontend decides how to treat low- vs. high-confidence ones (see the
# highConfidence flag added later).
MIN_CONFIDENCE = 0.10

# At or above this confidence, a detection is treated as "high confidence" and
# the frontend pre-checks it for the contributor. Lower-confidence detections
# are still shown, just not pre-checked (per the project plan's trust rule).
HIGH_CONFIDENCE = 0.85


def detect(image_path):
    """Run YOLO-World on an image and return a list of detection dicts.

    Each dict is shaped for the frontend. Detections whose prompt has no
    mapped accessibility feature are skipped, so the frontend only receives
    keys it knows how to display.
    """
    # predict() returns a list of Results (one per image). We pass one image,
    # so we take results[0]. verbose=False keeps the console quiet.
    results = _model.predict(image_path, conf=MIN_CONFIDENCE, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        # box.cls is the index into PROMPTS; look up the human-readable prompt.
        class_index = int(box.cls)
        prompt = result.names[class_index]

        feature = to_feature(prompt)
        if feature is None:
            # We don't have a frontend feature for this label — skip it.
            continue

        # box.xyxy is [x1, y1, x2, y2] in original image pixels. The frontend
        # wants {x, y, width, height}, so convert.
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bounding_box = {
            "x": round(x1),
            "y": round(y1),
            "width": round(x2 - x1),
            "height": round(y2 - y1),
        }

        confidence = round(float(box.conf), 2)
        detections.append(
            {
                "cocoLabel": prompt,
                "accessibilityFeature": feature,
                "confidence": confidence,
                "highConfidence": confidence >= HIGH_CONFIDENCE,
                "boundingBox": bounding_box,
            }
        )

    return detections
