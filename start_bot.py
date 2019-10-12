import numpy as np
import cv2
import pyscreenshot as ImageGrab

from template.template import screen_template
from utils import utils

while True:
    i = 2
    img = ImageGrab.grab(bbox=screen_template.to_tuple())  # x, y, x2, y2
    img_np = np.array(img)
    resized = utils.resize_to_rectangle(img_np, screen_template)
    cv2.imshow("frame", resized)
    key = cv2.waitKey(1)
    if key == 27:
        break
