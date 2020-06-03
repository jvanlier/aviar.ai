"""Inference for Birdcam."""
from pathlib import Path
import platform

import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

from aviar.env import TFLITE_MODEL_PATH, KERAS_RESCALE, KERAS_MEAN, KERAS_IMG_SIZE
from aviar.infer.abstract import AbstractInference


EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


class TfLiteInference(AbstractInference):
    def __init__(self):
        path = Path(TFLITE_MODEL_PATH).expanduser()

        if not path.exists():
            raise FileNotFoundError(Path)

        self.interpreter = tflite.Interpreter(
            model_path=str(path),
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB, {})
            ])
        self.interpreter.allocate_tensors()

    def predict(self, img: np.ndarray) -> float:
        """Return probability of BirdHome."""
        img_small = cv2.resize(img, KERAS_IMG_SIZE, interpolation=cv2.INTER_AREA)
        img_small = (img_small * KERAS_RESCALE) - KERAS_MEAN
        return self.classify_image(self.interpreter, img_small)

    @staticmethod
    def set_input_tensor(interpreter, input):
        input_details = interpreter.get_input_details()[0]
        tensor_index = input_details['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = input

    @staticmethod
    def classify_image(interpreter, input):
        TfLiteInference.set_input_tensor(interpreter, input)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = interpreter.get_tensor(output_details['index'])
        return output[0][0]
