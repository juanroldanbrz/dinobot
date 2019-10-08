import numpy as np
import cv2
import pyscreenshot as ImageGrab

while (True):
    i = 2
    img = ImageGrab.grab(bbox=(0, 180, 500, 400))  # x, y, x2, y2
    img_np = np.array(img)
    cv2.imshow("frame", img_np)
    key = cv2.waitKey(1)
    if key == 27:
        break
