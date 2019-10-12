import numpy as np
import cv2
import pyscreenshot as ImageGrab

from model.rectangle import Rectangle
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


def to_gray(img_np):
    return cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)


def crop_image(img_np, rectangle: Rectangle):
    return img_np[rectangle.y1:rectangle.y2, rectangle.x1:rectangle.x2].copy()
