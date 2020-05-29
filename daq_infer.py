#!/usr/bin/env python3
from time import sleep
import logging

from aviar.cam_interface import fetch_jpeg_as_array_cropped
from aviar.infer import FastaiInference


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s "
                           "[%(module)s/%(funcName)s]: %(message)s")


def main():
    inf = FastaiInference()

    while True:
        img = fetch_jpeg_as_array_cropped()
        import matplotlib.pyplot as plt
        plt.imsave("img.jpg", img)
        
        #img = plt.imread("/home/pi/labeled-sample/BirdRoaming/20200426T140201.jpeg")

        pred_bird_home = inf.predict(img)
        logging.info(f"BirdHome: {pred_bird_home:5.3f}")
        sleep(2)


if __name__ == "__main__":
    main()
