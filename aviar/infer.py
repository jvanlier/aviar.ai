"""Inference for Birdcam."""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from fastai.vision import load_learner, Image
import torch
import tensorflow as tf
import cv2

from .env import FASTAI_MODEL_PATH, KERAS_MODEL_PATH, KERAS_MEAN, KERAS_RESCALE, KERAS_IMG_SIZE


class AbstractInference(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, img: np.ndarray) -> float:
        """Return probability of BirdHome."""
        pass


class FastaiInference(AbstractInference):
    def __init__(self):
        path = Path(FASTAI_MODEL_PATH).expanduser()

        if not path.is_file():
            raise FileNotFoundError(path)

        self.learn = load_learner(path.parent, path.name)

    def predict(self, img: np.ndarray) -> float:
        """Return probability of BirdHome."""
        img_tensor = torch.from_numpy(img).float() / 255
        # MPL has colors last, while torch expects colors first. Swap:
        img_tensor = img_tensor.permute(2, 0, 1)
        img = Image(img_tensor)

        preds = self.learn.predict(img)
        pred_bird_home = preds[2][0]
        return pred_bird_home


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
