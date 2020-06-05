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
QUEUE_SIZE = 20
QUEUE_SIZE_STALL_TRIGGER = 6
SLEEP_S_INF_MANAGER = 1


class CameraReaderThread(Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def run(self):
        logging.info("CameraReaderThread starting")
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
                logging.info(f"Placed image on queue (Q size: {self.queue.qsize()})")

                # Do not sleep here to introduce artificial latency - it seems to cause read errors.
                # Instead, change frame rate on the camera's RTSP stream.

            cap.release()
            logging.warning(f"cap.isOpened() returned false - sleeping {SLEEP_S_CAP_CLOSED}"
                            "sec before reconnecting.")
            sleep(SLEEP_S_CAP_CLOSED)

    @staticmethod
    def is_gray(img):
        """Compare the first two channels to determine if the image is grayscale."""
        return np.all(img[:, :, 0] == img[:, :, 1])


class InferenceThread(Thread):
    def __init__(self, queue, thread_id):
        super().__init__()
        self.queue = queue
        self._inf_thread_id = thread_id
        self.killed = False

        self.inf = TfLiteInference()

    def run(self):
        while True:
            img = self.queue.get()
            pred_bird_home = self.inf.predict(img)
            bird_home = "Yes" if pred_bird_home > PROB_THRESHOLD else "No "
            logging.info(f"[InfThr {self._inf_thread_id:03d}] "
                         f"BirdHome: {bird_home}, p: {pred_bird_home:5.3f} "
                         f"(Q size: {self.queue.qsize()})")

            if self.killed:
                return


class InferenceManagerThread(Thread):
    """The inference thread tends to stall at some point. Probably some kind of bug in the TPU code.
    The workaround is to just restart it when that happens.
    A downside is that we might end up with a lot of stale threads.
    """
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self._inf_thread_id = 0

    def _start_inference(self):
        self._inf_thread_id += 1
        logging.info(f"InferenceManager starting InfThr {self._inf_thread_id}")
        self.inference_thread = InferenceThread(self.queue, self._inf_thread_id)
        self.inference_thread.start()

    def run(self):
        self._start_inference()
        while True:
            if self.queue.qsize() > QUEUE_SIZE_STALL_TRIGGER:
                logging.warning("Queue size exceeds tolerance "
                                f"(Q size: {self.queue.qsize()}, tolerance: {QUEUE_SIZE_STALL_TRIGGER}), "
                                f"assuming InfThr {self._inf_thread_id} has stalled")
                self.inference_thread.killed = True
                self._start_inference()

            sleep(SLEEP_S_INF_MANAGER)


def main():
    queue = Queue(maxsize=QUEUE_SIZE)
    cam = CameraReaderThread(queue)
    inf_mgr = InferenceManagerThread(queue)
    cam.start()
    inf_mgr.start()
    cam.join()
    inf_mgr.join()


if __name__ == "__main__":
    main()
