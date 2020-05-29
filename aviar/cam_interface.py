"""Talk to Birdcam and get usable JPEGs."""
import requests

from .env import JPG_URL


class FetchException(Exception):
    pass


def fetch_jpeg() -> bytes:
    """Fetch JPEG from camera using HTTP get.

    Returns:
        raw bytes, representing a JPEG

    Raises:
        FetchException in case of a non-200 status code.
    """
    req = requests.get(JPG_URL)
    if req.status_code == 200:
        return req.content
    else:
        raise FetchException(f"Got status code: {req.status_code}")


def fetch_jpeg_as_array_cropped():
    jpeg_bytes = fetch_jpeg()


    #TODO: 
    #- load as numpy array (RGB)
    #- crop
    #- unit test, mocking fetch_jpeg
    
