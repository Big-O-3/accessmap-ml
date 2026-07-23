"""
app.py — the web server. Wraps the detector in HTTP endpoints so the Node
backend can send a photo and get detections back.

Run it:
    source venv/bin/activate
    python app.py

Then in another terminal:
    curl http://localhost:5001/health
    curl -F "image=@samples/test.jpg" http://localhost:5001/analyze
"""

import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from detector import analyze as run_analyze

app = Flask(__name__)

# Allow browser-based frontends (the React app) to call this service directly.
# Open to all origins since this is a local dev / detection service with no
# sensitive data; tighten to specific origins if deployed.
CORS(app)

# Where uploaded photos are temporarily saved before analysis. Ignored by git.
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health():
    """Simple check so you can confirm the server is running."""
    return jsonify({"status": "ok"})


@app.post("/analyze")
def analyze():
    """Accept an uploaded image and return detected accessibility features.

    Expects multipart/form-data with a file field named "image".
    Returns: { "detections": [ ... ] }
    """
    # request.files holds uploaded files keyed by their form field name.
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (use form field 'image')."}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "Empty filename — no file was selected."}), 400

    # Save the upload to disk so the detector can read it, then analyze it.
    save_path = os.path.join(UPLOAD_DIR, image.filename)
    image.save(save_path)

    result = run_analyze(save_path)
    detections = result["detections"]
    return jsonify(
        {
            "detections": detections,
            "altTextSuggestion": build_alt_text(detections),
            "isVenue": result["isVenue"],
            "framingHint": result["framingHint"],
        }
    )


def build_alt_text(detections):
    """Make a short plain-English summary for screen-reader users.

    Example: "Detected: door, chair." Uses the human-readable prompt
    (cocoLabel) and lists each detected thing once.
    """
    if not detections:
        return "No accessibility features detected."

    # Keep unique labels in the order they were first seen.
    labels = []
    for d in detections:
        if d["cocoLabel"] not in labels:
            labels.append(d["cocoLabel"])
    return "Detected: " + ", ".join(labels) + "."


if __name__ == "__main__":
    # Port 5001 avoids clashing with the Node backend (commonly on 3000/5000).
    app.run(port=5001, debug=True)
