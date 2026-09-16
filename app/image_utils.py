"""Image preprocessing matching improving-model.ipynb's training pipeline
(cell 40's load_image function): decode -> resize to 224x224 -> float32,
no manual rescaling (MobileNetV3's normalization is baked into the model
itself, so tf.keras.applications.mobilenet_v3.preprocess_input is a
no-op in the original notebook too).

Uses PIL instead of tf.image so this module has no TensorFlow
dependency (see app/model.py for why).
"""

import io

import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)
