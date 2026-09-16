"""Loads B4_Hybrid_Frozen_BASELINE.keras and runs predictions.

The model takes three named inputs (side_image, rear_image: 224x224x3
float32; geometry: 13 normalized floats) and outputs a single
weight_kg value. See app/geometry.py for how geometry is derived and
normalized, and app/image_utils.py for image preprocessing.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "B4_Hybrid_Frozen_BASELINE.keras"

_model: tf.keras.Model | None = None


def load_model() -> tf.keras.Model:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def predict_weight(
    side_image: np.ndarray,
    rear_image: np.ndarray,
    normalized_geometry: list[float],
) -> float:
    model = load_model()

    inputs = {
        "side_image": np.expand_dims(side_image, axis=0),
        "rear_image": np.expand_dims(rear_image, axis=0),
        "geometry": np.array([normalized_geometry], dtype=np.float32),
    }

    prediction = model.predict(inputs, verbose=0)
    return float(np.reshape(prediction, (-1,))[0])
