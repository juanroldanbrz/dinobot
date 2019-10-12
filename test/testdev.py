import cv2
import numpy as np

from utils import utils

utils.snapshot_to_file('test.jpg')
#
# img = cv2.imread('snapshots/game_over.png', 1)
# img_np = np.array(img)
# utils.display_rectangle(img_np, game_over_template)
# cv2.imshow('test', img_np)
# cv2.waitKey(0)