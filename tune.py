"""
tune.py — a measurement harness for tuning Grounding DINO prompts and thresholds.

Detection tuning is empirical: you change the prompt phrasing or confidence
thresholds, then SEE what actually gets detected on a fixed set of images.
Guessing doesn't work; measuring does.

Usage:
    python tune.py            # run the candidate config below on all test images
    python tune.py --download # (re)download the test image set into tune_images/

Edit CANDIDATE_PROMPT and the thresholds below, re-run, and compare the tables.
When a config looks good, copy the winning prompt into features.py (PROMPT_TEXT)
and the thresholds into detector.py (MIN_CONFIDENCE / TEXT_THRESHOLD).
"""

import os
import sys
import urllib.request

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# --- The fixed test set --------------------------------------------------------
# A handful of realistic venue photos. Add your own venue photos to tune_images/
# to test against real data.
TEST_IMAGES = {
    "ramp_entrance.jpg": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800&q=70",
    "storefront_door.jpg": "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=800&q=70",
    "cafe_seating.jpg": "https://images.unsplash.com/photo-1590725140246-20acdee442be?w=800&q=70",
    "stairs.jpg": "https://images.unsplash.com/photo-1555529669-e69e7aa0ba9a?w=800&q=70",
    "restaurant_interior.jpg": "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=800&q=70",
}

IMAGE_DIR = "tune_images"
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# --- The candidate config to test ---------------------------------------------
# Edit these, re-run, and compare. This is the whole tuning loop.
# Grounding DINO expects concepts separated by periods, lowercase.
CANDIDATE_PROMPT = (
    "wheelchair ramp. ramp. door. entrance. "
    "stairs. steps. staircase. handrail. "
    "chair. bench. table. toilet. sink."
)

CANDIDATE_BOX_THRESHOLD = 0.30   # minimum box confidence
CANDIDATE_TEXT_THRESHOLD = 0.30  # minimum text-match strength


def download_images():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    for name, url in TEST_IMAGES.items():
        path = os.path.join(IMAGE_DIR, name)
        if os.path.exists(path):
            continue
        print(f"downloading {name} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "accessmap-tune/1.0"})
        with urllib.request.urlopen(req) as r, open(path, "wb") as f:
            f.write(r.read())


def run():
    if not os.path.isdir(IMAGE_DIR):
        print("No test images yet — run: python tune.py --download")
        return

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)

    print(f"\nPrompt: {CANDIDATE_PROMPT}")
    print(f"box_threshold={CANDIDATE_BOX_THRESHOLD}, text_threshold={CANDIDATE_TEXT_THRESHOLD}\n")
    print(f"{'image':<26}{'detections (label @ conf)'}")
    print("-" * 80)

    for name in sorted(os.listdir(IMAGE_DIR)):
        image = Image.open(os.path.join(IMAGE_DIR, name)).convert("RGB")
        inputs = processor(images=image, text=CANDIDATE_PROMPT, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=CANDIDATE_BOX_THRESHOLD,
            text_threshold=CANDIDATE_TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]

        hits = sorted(
            (f"{lbl}@{float(s):.2f}" for lbl, s in zip(result["labels"], result["scores"])),
            key=lambda s: float(s.split("@")[1]),
            reverse=True,
        )
        print(f"{name:<26}{', '.join(hits) if hits else 'NONE'}")
    print("-" * 80)
    print("Tip: find thresholds where real features appear but noise/duplicates don't.")


if __name__ == "__main__":
    if "--download" in sys.argv:
        download_images()
    run()
