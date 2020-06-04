#!/usr/bin/env python3
from time import sleep
import logging
from queue import Queue
from threading import Thread

import cv2
import numpy as np

from aviar.cam_interface import apply_roi
from aviar.infer.tflite import TfLiteInference
from aviar.env import RTSP_URL


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s "
                           "[%(module)s/%(funcName)s]: %(message)s")


SLEEP_S_READ_ERROR = 5
SLEEP_S_GRAYSCALE = 60
SLEEP_S_CAP_CLOSED = 5
PROB_THRESHOLD = .5


class CameraReaderThread(Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        while True:
            cap = cv2.VideoCapture(RTSP_URL)
            while(cap.isOpened()):
                ret, img_bgr = cap.read()
                if not ret:
                    logging.error(f"Coud not read frame. Sleeping {SLEEP_S_READ_ERROR} sec before retry")
                    sleep(SLEEP_S_READ_ERROR)
                    continue

                if self.is_gray(img_bgr):
                    logging.info(f"Got grayscale image - not supported. Sleeping {SLEEP_S_GRAYSCALE} sec.")
                    sleep(SLEEP_S_GRAYSCALE)
                    continue

                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img = apply_roi(img)
                # plt.imsave("last-img.jpg", img)

                self.queue.put(img)
                # sleep(1)

            cap.release()
            logging.warning(f"cap.isOpened() returned false - sleeping {SLEEP_S_CAP_CLOSED}"
                            "sec before reconnecting.")
            sleep(SLEEP_S_CAP_CLOSED)

    @staticmethod
    def is_gray(img):
        """Compare the first two channels to determine if the image is grayscale."""
        return np.all(img[:, :, 0] == img[:, :, 1])


class InferenceThread(Thread):
    def __init__(self, queue):
        super().__init__()
        self.inf = TfLiteInference()
        self.queue = queue

    def run(self):
        while True:
            img = self.queue.get()
            pred_bird_home = self.inf.predict(img)
            bird_home = "Yes" if pred_bird_home > PROB_THRESHOLD else "No "
            logging.info(f"BirdHome: {bird_home}, p: {pred_bird_home:5.3f}")


def main():
    queue = Queue(maxsize=20)
    cam = CameraReaderThread(queue)
    inf = InferenceThread(queue)
    cam.start()
    inf.start()
    cam.join()
    inf.join()


if __name__ == "__main__":
    main()
