import numpy as np


JPG_URL = "http://192.168.2.50/snap.jpeg"

ROI_Y_MIN = 100
ROI_Y_MAX = 1080
ROI_X_MIN = 400
ROI_X_MAX = 1200

FASTAI_MODEL_PATH = "~/models/model_v2.1-frozen-epoch04-thaw-epoch06"

KERAS_MODEL_PATH = "/home/pi/models/export"
KERAS_IMG_SIZE = (336, 336)
KERAS_RESCALE = 1/128
KERAS_MEAN = np.array([[[0.74136317, 0.73086095, 0.74473]]], dtype=np.float32)
