"""Talk to Birdcam and get usable JPEGs."""
from io import BytesIO
#from urllib import URLError
import urllib.request as request

import numpy as np
import matplotlib.pyplot as plt

from .env import JPG_URL, ROI_X_MIN, ROI_X_MAX, ROI_Y_MIN, ROI_Y_MAX


class FetchException(Exception):
    pass


def _fetch_jpeg() -> bytes:
    """Fetch JPEG from camera using HTTP get.

    Returns:
        raw bytes, representing a JPEG

    Raises:
        FetchException in case of protocol errors (urllib.URLError) or a
            non-200 status code.
    """
    with request.urlopen(JPG_URL) as url:
        code = url.getcode()
        if code != 200:
            raise FetchException(f"Got a non-200 status code: {code}")
        return url.read()


def fetch_jpeg_as_array_cropped() -> np.ndarray:
    """Fetch JPEG from camera using HTTP get, loads using matplotlib and
    crops to region of interest.

    Returns:
        numpy array

    Raises:
        FetchException in case of protocol errors (urllib.URLError) or a
            non-200 status code.
    """
    jpeg_bytes = _fetch_jpeg()
    bio = BytesIO(jpeg_bytes)

    img = plt.imread(bio, format="jpeg")
    return img[ROI_Y_MIN:ROI_Y_MAX, ROI_X_MIN:ROI_X_MAX]
