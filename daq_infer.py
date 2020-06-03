#!/usr/bin/env python3
from time import sleep
import logging

import cv2
import numpy as np

from aviar.cam_interface import apply_roi
from aviar.infer.tflite import TfLiteInference


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s "
                           "[%(module)s/%(funcName)s]: %(message)s")


def is_gray(img):
    return np.all(img[:, :, 0] == img[:, :, 1])


def main():
    inf = TfLiteInference()

    while True:
        cap = cv2.VideoCapture("rtsp://192.168.2.50:554/s0")
        while(cap.isOpened()):
            ret, img_bgr = cap.read()
            if not ret:
                logging.error("Coud not read frame. Sleeping 60 sec before retry")
                sleep(60)
                break

            if is_gray(img_bgr):
                logging.info("Got grayscale image - not supported. Sleeping 60 sec.")
                sleep(60)
                continue

            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = apply_roi(img)
            # plt.imsave("last-img.jpg", img)

            pred_bird_home = inf.predict(img)
            bird_home = "Yes" if pred_bird_home > .5 else "No "
            logging.info(f"BirdHome: {bird_home}, p: {pred_bird_home:5.3f}")
            # sleep(1)

        cap.release()
        logging.warning("cap.isOpened() returned false - sleeping 60 sec before reconnecting.")
        sleep(60)


if __name__ == "__main__":
    main()
