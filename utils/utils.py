import numpy as np
import cv2
import pyscreenshot as ImageGrab

from template.template import screen_template, game_over_template


def snapshot_to_file(file_name):
    img = ImageGrab.grab(bbox=screen_template.to_tuple())  # x, y, x2, y2
    img_np = np.array(img)
    resized = resize_to_rectangle(img_np, screen_template)
    cv2.imwrite(file_name, resized)


def resize_to_rectangle(img_np, rectangle):
    return cv2.resize(img_np, (rectangle.width(), rectangle.height()))


def display_rectangle(img_np, rectangle):
    points = rectangle.to_points()
    cv2.rectangle(img_np, points[0], points[1], color=255)
