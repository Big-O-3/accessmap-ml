"""
features.py — the "dictionary" that connects the AI model to the frontend.

We use Grounding DINO, an open-vocabulary detector that takes a text prompt of
things to look for (phrases separated by periods) and returns boxes with the
matched phrase as a label.

Two pieces live here:
  1. PROMPT_TEXT  — the phrase string we feed Grounding DINO.
  2. KEYWORD_MAP  — maps a keyword found in a returned label to a frontend
                    feature key. Grounding DINO labels can be multi-word or
                    partial (e.g. "a handrail", "stairs steps"), so we match by
                    keyword rather than exact string.

IMPORTANT: the feature keys below MUST match the frontend's list in
accessmap-frontend-/src/lib/features.js exactly.
"""

# The prompt fed to Grounding DINO. Lowercase, each concept ended with a period
# (its expected format). Multiple phrasings improve recall.
#
# Note: "accessible parking sign" was removed after tuning — it false-fired
# (0.3-0.4 confidence) on photos containing no parking sign. Abstract/text-based
# concepts are unreliable for this detector; concrete objects work best.
PROMPT_TEXT = (
    "wheelchair ramp. ramp. door. entrance. "
    "stairs. steps. staircase. "
    "handrail. "
    "chair. bench. table. "
    "toilet. sink."
)

# Maps a keyword (searched inside a returned label) -> frontend feature key.
# Order matters: the FIRST keyword found in a label wins, so put more specific
# terms first (e.g. "ramp" before generic entrance terms). Frontend keys:
#   entrance_detected, restroom_available, parking_area,
#   seating_available, indoor_seating, stairs_present
KEYWORD_MAP = [
    ("ramp", "entrance_detected"),
    ("stairs", "stairs_present"),   # barrier — checked before generic words
    ("steps", "stairs_present"),
    ("staircase", "stairs_present"),
    ("handrail", "stairs_present"), # handrails accompany steps/ramps; flag as a barrier cue
    ("door", "entrance_detected"),
    ("entrance", "entrance_detected"),
    ("chair", "seating_available"),
    ("bench", "seating_available"),
    ("table", "seating_available"),
    ("toilet", "restroom_available"),
    ("sink", "restroom_available"),
]


def to_feature(label):
    """Map a Grounding DINO label to a frontend feature key, or None.

    Matches by keyword so partial/multi-word labels ("a handrail", "stairs
    steps") still resolve. Returns None for anything unmapped so the detector
    can skip it.
    """
    text = label.lower()
    for keyword, feature in KEYWORD_MAP:
        if keyword in text:
            return feature
    return None
