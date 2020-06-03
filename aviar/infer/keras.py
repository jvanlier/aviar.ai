from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from aviar.env import KERAS_MODEL_PATH, KERAS_RESCALE, KERAS_MEAN, KERAS_IMG_SIZE
from aviar.infer.abstract import AbstractInference


class KerasInference(AbstractInference):
    def __init__(self):
        path = Path(KERAS_MODEL_PATH).expanduser()

        if not path.exists():
            raise FileNotFoundError(Path)

        self.model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    def predict(self, img: np.ndarray) -> float:
        """Return probability of BirdHome."""
        img_small = cv2.resize(img, KERAS_IMG_SIZE)
        img_small = (img_small * KERAS_RESCALE) - KERAS_MEAN
        img_small_batch = np.expand_dims(img_small, 0)

        preds = self.model.predict(img_small_batch)

        return preds[0][0]
