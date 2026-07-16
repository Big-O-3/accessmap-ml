"""
tune.py — a measurement harness for tuning YOLO-World prompts and threshold.

Detection tuning is empirical: you change the prompt phrasing or confidence
floor, then SEE what actually gets detected on a fixed set of images. Guessing
doesn't work; measuring does.

Usage:
    python tune.py            # run the candidate config below on all test images
    python tune.py --download # (re)download the test image set into tune_images/

Edit CANDIDATE_PROMPTS and CANDIDATE_CONF below, re-run, and compare the tables.
When a config looks good, copy the winning prompts into features.py and the
threshold into detector.py.
"""

import os
import sys
import urllib.request

from ultralytics import YOLOWorld

# --- The fixed test set --------------------------------------------------------
# A handful of realistic venue photos, each labeled with what a human sees so we
# can judge whether detection is finding the right things. Add your own venue
# photos to tune_images/ to test against real data.
TEST_IMAGES = {
    "ramp_entrance.jpg": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800&q=70",
    "storefront_door.jpg": "https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=800&q=70",
    "cafe_seating.jpg": "https://images.unsplash.com/photo-1590725140246-20acdee442be?w=800&q=70",
    "stairs.jpg": "https://images.unsplash.com/photo-1555529669-e69e7aa0ba9a?w=800&q=70",
    "restaurant_interior.jpg": "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=800&q=70",
}

IMAGE_DIR = "tune_images"

# --- The candidate config to test ---------------------------------------------
# Edit these two, re-run, and compare. This is the whole tuning loop.
CANDIDATE_PROMPTS = [
    "wheelchair ramp",
    "ramp",
    "door",
    "glass door",
    "stairs",
    "steps",
    "handrail",
    "accessible parking sign",
    "chair",
    "bench",
    "table",
    "toilet",
]

CANDIDATE_CONF = 0.05  # low floor so we can SEE everything, then decide where to cut


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

    model = YOLOWorld("yolov8s-world.pt")
    model.set_classes(CANDIDATE_PROMPTS)

    print(f"\nPrompts ({len(CANDIDATE_PROMPTS)}): {CANDIDATE_PROMPTS}")
    print(f"Confidence floor: {CANDIDATE_CONF}\n")
    print(f"{'image':<26}{'detections (prompt @ conf)'}")
    print("-" * 80)

    images = sorted(os.listdir(IMAGE_DIR))
    for name in images:
        path = os.path.join(IMAGE_DIR, name)
        result = model.predict(path, conf=CANDIDATE_CONF, verbose=False)[0]
        hits = []
        for box in result.boxes:
            label = result.names[int(box.cls)]
            hits.append(f"{label}@{float(box.conf):.2f}")
        # Sort by confidence (highest first) for readability.
        hits.sort(key=lambda s: float(s.split("@")[1]), reverse=True)
        print(f"{name:<26}{', '.join(hits) if hits else 'NONE'}")
    print("-" * 80)
    print("Tip: look for the confidence where real features appear but junk doesn't.")


if __name__ == "__main__":
    if "--download" in sys.argv:
        download_images()
    run()
