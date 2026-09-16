"""FastAPI backend for the B4 hybrid cattle weight model
(side image + rear image + body keypoints -> predicted weight in kg).

This is a standalone service, independent from any existing deployed
prediction API. It is not wired into the Flutter app yet.

POST /predict expects multipart/form-data:
    side_image:     file  (JPEG/PNG)
    rear_image:     file  (JPEG/PNG)
    side_keypoints: str   JSON array of 9 [x, y] pixel points, in order:
                          wither, pinbone, shoulderbone,
                          front_girth_top, front_girth_bottom,
                          rear_girth_top, rear_girth_bottom,
                          height_top, height_bottom
    rear_keypoints: str   JSON array of 4 [x, y] pixel points, in order:
                          top, bottom, left, right

Keypoints must be in pixel coordinates of the respective uploaded image
as-is (not resized, not normalized 0-1) -- see app/geometry.py for why.
"""

import json
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.geometry import (
    GeometryError,
    REAR_KEYPOINT_COUNT,
    SIDE_KEYPOINT_COUNT,
    build_geometry_features,
    normalize_geometry,
)
from app.image_utils import preprocess_image_bytes
from app.model import load_model, predict_weight

logger = logging.getLogger("cattle_weight_hybrid_api")

app = FastAPI(title="Cattle Weight Hybrid API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    predicted_weight_kg: float
    geometry_features: dict[str, float]


@app.on_event("startup")
def _warm_up_model() -> None:
    load_model()
    logger.info("Model loaded and ready.")


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "model": "B4_Hybrid_Frozen_BASELINE"}


def _parse_points(raw: str, expected_count: int, field_name: str) -> list[tuple[float, float]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is not valid JSON: {exc}",
        ) from exc

    if not isinstance(parsed, list) or len(parsed) != expected_count:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field_name} must be a JSON array of {expected_count} "
                f"[x, y] pairs."
            ),
        )

    try:
        return [(float(p[0]), float(p[1])) for p in parsed]
    except (TypeError, ValueError, IndexError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} contains an invalid [x, y] pair: {exc}",
        ) from exc


@app.post("/predict", response_model=PredictResponse)
async def predict(
    side_image: UploadFile = File(...),
    rear_image: UploadFile = File(...),
    side_keypoints: str = Form(...),
    rear_keypoints: str = Form(...),
) -> PredictResponse:
    side_points = _parse_points(side_keypoints, SIDE_KEYPOINT_COUNT, "side_keypoints")
    rear_points = _parse_points(rear_keypoints, REAR_KEYPOINT_COUNT, "rear_keypoints")

    try:
        geometry_features = build_geometry_features(side_points, rear_points)
        normalized_geometry = normalize_geometry(geometry_features)
    except GeometryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    side_bytes = await side_image.read()
    rear_bytes = await rear_image.read()

    if not side_bytes or not rear_bytes:
        raise HTTPException(status_code=400, detail="One of the uploaded images is empty.")

    try:
        side_tensor = preprocess_image_bytes(side_bytes)
        rear_tensor = preprocess_image_bytes(rear_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode an image: {exc}") from exc

    weight_kg = predict_weight(side_tensor, rear_tensor, normalized_geometry)

    return PredictResponse(
        predicted_weight_kg=round(weight_kg, 2),
        geometry_features=geometry_features,
    )
