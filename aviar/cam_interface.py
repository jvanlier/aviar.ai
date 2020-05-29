"""Talk to Birdcam and get usable JPEGs."""
from io import BytesIO
import urllib

import matplotlib.pyplot as plt

from .env import JPG_URL, ROI_X_MIN, ROI_X_MAX, ROI_Y_MIN, ROI_Y_MAX


class FetchException(Exception):
    pass


def fetch_jpeg() -> bytes:
    """Fetch JPEG from camera using HTTP get.

    Returns:
        raw bytes, representing a JPEG

    Raises:
        FetchException in case of a non-200 status code.
    """
    with urllib.request.urlopen(JPG_URL) as url:
        return url.read()


def fetch_jpeg_as_array_cropped():
    jpeg_bytes = fetch_jpeg()
    bio = BytesIO(jpeg_bytes)

    img = plt.imread(bio, format="jpeg")
    return img[ROI_Y_MIN:ROI_Y_MAX, ROI_X_MIN:ROI_X_MAX]
