"""
features.py — the "dictionary" that connects the AI model to the frontend.

YOLO-World is an *open-vocabulary* detector: instead of a fixed list of classes,
we hand it plain-English text prompts and it looks for those things in the photo.

Two pieces live here:
  1. PROMPTS      — the phrases we ask YOLO-World to look for.
  2. FEATURE_MAP  — turns each prompt into the exact feature key the frontend uses.

IMPORTANT: the feature keys below MUST match the frontend's list in
accessmap-frontend-/src/lib/features.js exactly. If a key is misspelled here, the
frontend will show the raw key (e.g. "entrance_detected") instead of a nice label
("Wheelchair accessible entrance").
"""

# The text phrases we ask the model to detect. Add more here as we learn what
# YOLO-World recognizes well. Each one must appear as a key in FEATURE_MAP below.
PROMPTS = [
    "wheelchair ramp",
    "door",
    "stairs",
    "accessible parking sign",
    "chair",
    "toilet",
]

# Maps a detected prompt -> the frontend accessibility feature key.
# Frontend keys (from features.js):
#   entrance_detected, restroom_available, parking_area,
#   seating_available, indoor_seating, stairs_present
FEATURE_MAP = {
    "wheelchair ramp": "entrance_detected",
    "door": "entrance_detected",
    "stairs": "stairs_present",          # a barrier, but still worth surfacing
    "accessible parking sign": "parking_area",
    "chair": "seating_available",
    "toilet": "restroom_available",
}


def to_feature(prompt):
    """Return the frontend feature key for a detected prompt, or None if unmapped.

    Returning None lets the detector skip anything we don't have a feature for,
    so the frontend only ever receives keys it knows how to display.
    """
    return FEATURE_MAP.get(prompt)
