"""Inference for Birdcam."""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from fastai.vision import load_learner, Image
import torch

from .env import MODEL_PATH


class AbstractInference(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, img: np.ndarray) -> float:
        """Return probability of BirdHome."""
        pass


class FastaiInference(AbstractInference):
    def __init__(self):
        path = Path(MODEL_PATH).expanduser()

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
