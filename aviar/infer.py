"""Inference for Birdcam."""
from abc import ABC, abstractmethod
from pathlib import Path
import platform

from PIL import Image
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

from . import classify
from .env import TFLITE_MODEL_PATH, KERAS_RESCALE, KERAS_MEAN, KERAS_IMG_SIZE


EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


class AbstractInference(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, img: np.ndarray) -> float:
        """Return probability of BirdHome."""
        pass


class TfLiteInference(AbstractInference):
    def __init__(self):
        path = Path(TFLITE_MODEL_PATH).expanduser()

        if not path.exists():
            raise FileNotFoundError(Path)

        self.model = tflite.Interpreter(
            model_path=str(path),
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB, {})
            ])
        self.model.allocate_tensors()

    def predict(self, img: np.ndarray) -> float:
        """Return probability of BirdHome."""
#        img_small = cv2.resize(img, KERAS_IMG_SIZE, interpolation=cv2.INTER_AREA)
#        img_small = (img_small * KERAS_RESCALE) - KERAS_MEAN
 #       classify.set_input(self.model, img_small)

        image = Image.fromarray(img).convert('RGB').resize(KERAS_IMG_SIZE, Image.ANTIALIAS)
        classify.set_input(self.model, image)

        self.model.invoke()
        preds = classify.get_output(self.model, 1, 0)

        return preds[0].score
