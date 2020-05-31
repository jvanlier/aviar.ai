from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt

import aviar.cam_interface as victim


def _mock_fetch_jpeg():
    bio = BytesIO()
    all_white = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    plt.imsave(bio, all_white, format="jpeg")
    return bio.getvalue()


def test_fetch_jpeg_as_array_cropped(monkeypatch):
    monkeypatch.setattr(victim, "_fetch_jpeg", _mock_fetch_jpeg) 

    img = victim.fetch_jpeg_as_array_cropped()

    assert img.shape == (980, 800, 3)
