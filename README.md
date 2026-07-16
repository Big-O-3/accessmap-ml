# accessmap-ml

The AI vision service for AccessMap. It takes a venue photo and returns a list of
detected accessibility features (ramps, doors, stairs, chairs, etc.) with confidence
scores and bounding boxes, using the open-vocabulary **Grounding DINO** model. The
Node/Express backend calls this service over HTTP, and the React frontend draws the
detections on the photo.

## How it works

- **Grounding DINO** is an *open-vocabulary* object detector: instead of a fixed class
  list, we give it plain-English prompts (e.g. `"wheelchair ramp"`, `"door"`, `"stairs"`)
  and it finds those things. The prompt and how detected labels map to the frontend's
  feature keys live in [`features.py`](features.py).
- [`detector.py`](detector.py) loads the model once and exposes `detect(image_path)`,
  which returns detections shaped exactly like the frontend expects.
- [`app.py`](app.py) wraps that in a small Flask server with two endpoints.

> **Why Grounding DINO and not YOLO-World?** We first used YOLO-World, but it has a
> blind spot for architectural accessibility features — it could not detect ramps,
> stairs, or doors at any confidence, even with the larger `yolov8x-world` model (it
> only reliably saw furniture). Grounding DINO reliably surfaces ramps, handrails,
> stairs, and doors, which is essential for the core accessibility use case. See the
> decision log in the project plan for the full rationale.

## Setup

```bash
cd accessmap-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **First run note:** the first detection downloads the Grounding DINO weights
> (`IDEA-Research/grounding-dino-tiny`, ~700 MB) into the Hugging Face cache. That's a
> one-time download; it's cached afterward. CPU is fine for local development; the first
> analysis after startup takes ~5–10 s while the model loads, then it's faster.

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

Run the unit + integration tests (they stub the model so no weights are loaded):

```bash
pytest -v
```

Tune detection prompts/thresholds against a fixed set of venue photos:

```bash
python tune.py --download   # first time: fetch the test images
python tune.py              # run the candidate config and print what it detects
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

- `cocoLabel` — the Grounding DINO label that matched (kept as `cocoLabel` for frontend
  compatibility).
- `accessibilityFeature` — a frontend feature key (must match
  `accessmap-frontend-/src/lib/features.js`).
- `confidence` — 0.0–1.0.
- `highConfidence` — `true` when confidence ≥ 0.5; the frontend pre-checks these.
  Lower-confidence detections are still returned, just not pre-checked.
- `boundingBox` — in original image pixels, `{ x, y, width, height }`.

## Files

| File | Purpose |
|---|---|
| `features.py` | Detection prompt + mapping of detected labels to frontend feature keys |
| `detector.py` | Loads Grounding DINO and runs `detect()` |
| `app.py` | Flask server: `GET /health`, `POST /analyze` |
| `tune.py` | Measurement harness for tuning prompts/thresholds against test images |
| `test_detect.py` | Manual smoke check: runs the detector on the sample image (loads real weights) |
| `test_features.py` / `test_detector.py` / `test_app.py` | `pytest` suites; stub the model so no weights load |
| `samples/test.jpg` | Sample photo for testing |

## Notes

- Feature keys in `features.py` must stay identical to the frontend's `features.js` — a
  typo means the frontend shows a raw key instead of a friendly label.
- Grounding DINO scores are tuned in `detector.py` via `MIN_CONFIDENCE = 0.30` and
  `TEXT_THRESHOLD = 0.30`, chosen with `tune.py` to keep real features (ramps, handrails,
  stairs, seating) while cutting noise and duplicate labels. Adjust the prompt in
  `features.py` and re-run `tune.py` as you learn what the model recognizes well.
- Detection has known limits: plain doors and some ramps are hard, so results lean on
  handrails/stairs/seating. A human still confirms every detection (per the project's
  trust model), so the model is a starting point, not the authority.
- Backend + Cloudinary wiring is intentionally out of scope here — this service takes a
  raw uploaded file so the ML can be tested on its own.
