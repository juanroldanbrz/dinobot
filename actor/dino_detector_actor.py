from typing import List

import cv2
import numpy as np

from model.actor import Actor
from model.rectangle import Rectangle
from template.template import screen_template, enemy_segment_template, dino_segment_template
from utils import utils
from utils.assertions import assert_rectangle_shape, assert_gray_img


def find_dino(full_gray_np) -> List[Rectangle]:
    # Assert
    assert_rectangle_shape(full_gray_np, screen_template, f'Shape of the image should be {screen_template.shape()}')
    assert_gray_img(full_gray_np)

    roi_gray_np = utils.crop_image(full_gray_np, dino_segment_template)
    _, mask = cv2.threshold(roi_gray_np, 200, 240, cv2.THRESH_BINARY_INV)

    kernel = np.ones((10, 10), np.uint8)
    dilation = cv2.dilate(mask, kernel)

    _, contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    to_return = list()

    if len(contours) != 1:
        return []
    else:
        x, y, w, h = cv2.boundingRect(contours[0])
        rectangle = Rectangle(x, y, x + w, y + h)
        return rectangle.relativize_from(dino_segment_template)