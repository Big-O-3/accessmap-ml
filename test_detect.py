"""
test_detect.py — quick sanity check for the detector, no web server needed.

Run it:
    source venv/bin/activate
    python test_detect.py

It runs the model on the sample image and prints the detections as JSON.
"""

import json

from detector import detect

if __name__ == "__main__":
    results = detect("samples/test.jpg")
    print(json.dumps(results, indent=2))
    print(f"\n{len(results)} detection(s) found.")
