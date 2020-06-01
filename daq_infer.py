#!/usr/bin/env python3
from time import sleep
import logging

import matplotlib.pyplot as plt

from aviar.cam_interface import fetch_jpeg_as_array_cropped
from aviar.infer import FastaiInference


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s "
                           "[%(module)s/%(funcName)s]: %(message)s")


def main():
    inf = FastaiInference()

    while True:
        img = fetch_jpeg_as_array_cropped()

        plt.imsave("last-img.jpg", img)

        pred_bird_home = inf.predict(img)
        bird_home = "Yes" if pred_bird_home > .5 else "No"
        logging.info(f"BirdHome: {bird_home}, p: {pred_bird_home:5.3f}")
        sleep(5)


if __name__ == "__main__":
    main()
