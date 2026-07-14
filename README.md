# accessmap-ml

The AI vision service for AccessMap. It takes a venue photo and returns a list of
detected accessibility features (ramps, doors, stairs, chairs, etc.) with confidence
scores and bounding boxes, using the open-vocabulary **YOLO-World** model. The
Node/Express backend calls this service over HTTP, and the React frontend draws the
detections on the photo.

## How it works

- **YOLO-World** is an *open-vocabulary* object detector: instead of a fixed class list,
  we give it plain-English prompts (e.g. `"wheelchair ramp"`, `"door"`, `"stairs"`) and
  it finds those things. The prompts and how they map to the frontend's feature keys live
  in [`features.py`](features.py).
- [`detector.py`](detector.py) loads the model once and exposes `detect(image_path)`,
  which returns detections shaped exactly like the frontend expects.
- [`app.py`](app.py) wraps that in a small Flask server with two endpoints.

## Setup

```bash
cd accessmap-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **First run note:** the first detection auto-downloads the model weights
> (`yolov8s-world.pt`, ~338 MB) and a CLIP dependency. That's a one-time download; it's
> cached afterward. CPU is fine for local development (a few seconds per image).

## Run the server

```bash
source venv/bin/activate
python app.py
```

The server starts on **http://localhost:5001**.

## Test it

Confirm the server is up:

```bash
curl http://localhost:5001/health
# {"status": "ok"}
```

Analyze a photo (a sample is included at `samples/test.jpg`):

```bash
curl -F "image=@samples/test.jpg" http://localhost:5001/analyze
```

Or test the model directly, without the server:

```bash
python test_detect.py
```

## Response shape

`POST /analyze` returns JSON shaped to match the frontend's `DetectionImage` component:

```json
{
  "detections": [
    {
      "cocoLabel": "door",
      "accessibilityFeature": "entrance_detected",
      "confidence": 0.94,
      "highConfidence": true,
      "boundingBox": { "x": 120, "y": 180, "width": 160, "height": 260 }
    }
  ],
  "altTextSuggestion": "Detected: door."
}
```

- `cocoLabel` — the text prompt that matched.
- `accessibilityFeature` — a frontend feature key (must match
  `accessmap-frontend-/src/lib/features.js`).
- `confidence` — 0.0–1.0.
- `highConfidence` — `true` when confidence ≥ 0.85; the frontend pre-checks these.
  Lower-confidence detections are still returned, just not pre-checked.
- `boundingBox` — in original image pixels, `{ x, y, width, height }`.

## Files

| File | Purpose |
|---|---|
| `features.py` | Detection prompts + mapping to frontend feature keys |
| `detector.py` | Loads YOLO-World and runs `detect()` |
| `app.py` | Flask server: `GET /health`, `POST /analyze` |
| `test_detect.py` | Runs the detector on the sample image (no server) |
| `samples/test.jpg` | Sample photo for testing |

## Notes

- Feature keys in `features.py` must stay identical to the frontend's `features.js` — a
  typo means the frontend shows a raw key instead of a friendly label.
- Open-vocabulary detections often score lower than fixed-class models, so
  `detector.py` uses a low confidence floor (`MIN_CONFIDENCE = 0.10`) to surface
  borderline features. Adjust the prompts in `features.py` as you learn what YOLO-World
  recognizes well.
- Backend + Cloudinary wiring is intentionally out of scope here — this service takes a
  raw uploaded file so the ML can be tested on its own.
