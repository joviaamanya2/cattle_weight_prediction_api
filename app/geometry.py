"""Keypoint -> geometry feature conversion.

Mirrors the exact formulas used in improving-model.ipynb (cell 36) to build
the 13 geometry features the model was trained on, and the exact
mean/std used to normalize them (cell 38 output, computed from the
training split only).

Side keypoints (9 points, in order):
    0 wither
    1 pinbone
    2 shoulderbone
    3 front girth top
    4 front girth bottom
    5 rear girth top
    6 rear girth bottom
    7 height top
    8 height bottom

Rear keypoints (4 points, in order):
    0 top
    1 bottom
    2 left
    3 right

All coordinates must be pixel coordinates on the ORIGINAL uploaded image
(not resized to 224x224, not normalized to 0-1) -- the training data's
geometry features are raw pixel distances, so the camera framing/distance
used to capture new photos should match the training dataset's setup for
predictions to be meaningful.
"""

from __future__ import annotations

import math

GEOMETRY_FEATURES = [
    "body_length_px",
    "height_px",
    "front_girth_px",
    "rear_girth_px",
    "shoulder_wither_px",
    "length_height_ratio",
    "front_girth_height_ratio",
    "rear_girth_height_ratio",
    "rear_front_girth_ratio",
    "shoulder_wither_height_ratio",
    "rear_height_px",
    "rear_width_px",
    "rear_width_height_ratio",
]

# Computed from the B4 training split only (notebook cell 38 output).
GEO_MEAN = {
    "body_length_px": 976.658409,
    "height_px": 962.098380,
    "front_girth_px": 477.036779,
    "rear_girth_px": 454.411439,
    "shoulder_wither_px": 328.980695,
    "length_height_ratio": 1.018055,
    "front_girth_height_ratio": 0.497276,
    "rear_girth_height_ratio": 0.473819,
    "rear_front_girth_ratio": 0.953083,
    "shoulder_wither_height_ratio": 0.343173,
    "rear_height_px": 1047.667841,
    "rear_width_px": 335.980884,
    "rear_width_height_ratio": 0.321366,
}

GEO_STD = {
    "body_length_px": 86.705979,
    "height_px": 85.892226,
    "front_girth_px": 37.305125,
    "rear_girth_px": 41.716930,
    "shoulder_wither_px": 35.457913,
    "length_height_ratio": 0.077829,
    "front_girth_height_ratio": 0.031316,
    "rear_girth_height_ratio": 0.049121,
    "rear_front_girth_ratio": 0.057582,
    "shoulder_wither_height_ratio": 0.035803,
    "rear_height_px": 117.237236,
    "rear_width_px": 47.997354,
    "rear_width_height_ratio": 0.034848,
}

SIDE_KEYPOINT_COUNT = 9
REAR_KEYPOINT_COUNT = 4


class GeometryError(ValueError):
    """Raised when keypoints cannot be turned into valid geometry features."""


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _safe_ratio(a: float, b: float) -> float:
    if abs(b) < 1e-8:
        raise GeometryError(
            "Cannot compute a ratio against a near-zero measurement; "
            "check the supplied keypoints."
        )
    return a / b


def side_geometry(points: list[tuple[float, float]]) -> dict[str, float]:
    if len(points) != SIDE_KEYPOINT_COUNT:
        raise GeometryError(
            f"Expected {SIDE_KEYPOINT_COUNT} side keypoints, got {len(points)}."
        )

    wither, pinbone, shoulder, front_top, front_bottom, rear_top, rear_bottom, \
        height_top, height_bottom = points

    body_length = _distance(wither, pinbone)
    shoulder_wither = _distance(shoulder, wither)
    front_girth = _distance(front_top, front_bottom)
    rear_girth = _distance(rear_top, rear_bottom)
    height = _distance(height_top, height_bottom)

    return {
        "body_length_px": body_length,
        "height_px": height,
        "front_girth_px": front_girth,
        "rear_girth_px": rear_girth,
        "shoulder_wither_px": shoulder_wither,
        "length_height_ratio": _safe_ratio(body_length, height),
        "front_girth_height_ratio": _safe_ratio(front_girth, height),
        "rear_girth_height_ratio": _safe_ratio(rear_girth, height),
        "rear_front_girth_ratio": _safe_ratio(rear_girth, front_girth),
        "shoulder_wither_height_ratio": _safe_ratio(shoulder_wither, height),
    }


def rear_geometry(points: list[tuple[float, float]]) -> dict[str, float]:
    if len(points) != REAR_KEYPOINT_COUNT:
        raise GeometryError(
            f"Expected {REAR_KEYPOINT_COUNT} rear keypoints, got {len(points)}."
        )

    top, bottom, left, right = points

    rear_height = _distance(top, bottom)
    rear_width = _distance(left, right)

    return {
        "rear_height_px": rear_height,
        "rear_width_px": rear_width,
        "rear_width_height_ratio": _safe_ratio(rear_width, rear_height),
    }


def build_geometry_features(
    side_points: list[tuple[float, float]],
    rear_points: list[tuple[float, float]],
) -> dict[str, float]:
    features = {}
    features.update(side_geometry(side_points))
    features.update(rear_geometry(rear_points))
    return features


def normalize_geometry(features: dict[str, float]) -> list[float]:
    return [
        (features[name] - GEO_MEAN[name]) / GEO_STD[name]
        for name in GEOMETRY_FEATURES
    ]
