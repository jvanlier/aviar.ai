"""Inference for Birdcam."""
from abc import ABC, abstractmethod

import numpy as np


class AbstractInference(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, img: np.ndarray) -> float:
        """Return probability of BirdHome."""
        pass
