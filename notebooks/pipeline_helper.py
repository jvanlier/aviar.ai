import imagehash as ihash
import cv2
from PIL import Image


ROI_Y_MIN = 100
ROI_Y_MAX = 1080
ROI_X_MIN = 400
ROI_X_MAX = 1200

HASH_SIZE = 13


def apply_roi(img):
    return img[ROI_Y_MIN:ROI_Y_MAX, ROI_X_MIN:ROI_X_MAX]


def img_roi_hash(img_path):
    img = cv2.imread(str(img_path))
    img_crop = apply_roi(img)

    hash_val = ihash.dhash(Image.fromarray(img_crop), hash_size=HASH_SIZE)

    return str(hash_val)
