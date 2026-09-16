# Cattle Weight Hybrid API

FastAPI backend for `B4_Hybrid_Frozen_BASELINE.keras`, the side-image +
rear-image + body-geometry cattle weight model from `improving-model.ipynb`.

This is a **new, standalone service**. It does not touch or replace the
existing production prediction API (`api-model-1-dzt0.onrender.com`) and
is not yet wired into the Flutter app.

## What the model needs

The model was trained on three inputs per prediction:

- a **side** photo of the animal
- a **rear** photo of the animal
- **13 geometry features** (body length, girths, height, etc.), computed
  as pixel distances between specific body keypoints, then normalized

There is no keypoint-detection model anywhere in the notebook or this
project -- the notebook only ever read keypoints from human-labeled COCO
annotation files that shipped with the training dataset. So this API
takes the 9 side keypoints and 4 rear keypoints as direct input (e.g. a
user tapping points on the photo in the app, or a keypoint detector you
add later) and computes the geometry features itself, using the same
formulas and the same training-set normalization stats as the notebook.

**Important caveat:** the geometry features are raw pixel distances, not
physically calibrated measurements. The training photos were captured
with a consistent camera setup/distance. Predictions will only be
meaningful if new photos are captured under similarly consistent framing
-- there's no correction for camera distance or zoom.

## API

### `GET /`
Health check.

### `POST /predict`
`multipart/form-data`:

| field            | type          | description                                   |
|------------------|---------------|------------------------------------------------|
| `side_image`     | file          | JPEG/PNG side photo                            |
| `rear_image`     | file          | JPEG/PNG rear photo                            |
| `side_keypoints` | string (JSON) | 9 `[x, y]` pixel points, see order below       |
| `rear_keypoints` | string (JSON) | 4 `[x, y]` pixel points, see order below       |

Keypoints are pixel coordinates on the uploaded image **as-is** (not
resized, not normalized to 0-1).

Side keypoint order:
`0 wither, 1 pinbone, 2 shoulderbone, 3 front girth top, 4 front girth
bottom, 5 rear girth top, 6 rear girth bottom, 7 height top, 8 height
bottom`

Rear keypoint order:
`0 top, 1 bottom, 2 left, 3 right`

Response:

```json
{
  "predicted_weight_kg": 187.42,
  "geometry_features": {
    "body_length_px": 981.2,
    "...": "..."
  }
}
```

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -F "side_image=@side.jpg" \
  -F "rear_image=@rear.jpg" \
  -F 'side_keypoints=[[100,50],[400,60],[120,80],[180,120],[180,300],[350,120],[350,300],[250,20],[250,320]]' \
  -F 'rear_keypoints=[[200,30],[200,350],[80,180],[320,180]]'
```

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Deploying (e.g. Render)

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Make sure `model/B4_Hybrid_Frozen_BASELINE.keras` is committed to the
  repo (it's ~6.4 MB, small enough for a normal git push, no LFS needed).
