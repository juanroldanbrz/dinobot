from typing import List, Tuple, Any

import cv2
import numpy as np

from model.rectangle import Rectangle
from template.template import screen_template, enemy_segment_template
from utils import utils
from utils.assertions import assert_rectangle_shape, assert_gray_img


def find_enemies(full_gray_np, game_phase=1) -> Tuple[List[Rectangle], Any]:
    # Assert
    assert_rectangle_shape(full_gray_np, screen_template, f'Shape of the image should be {screen_template.shape()}')
    assert_gray_img(full_gray_np)

    roi_gray_np = utils.crop_image(full_gray_np, enemy_segment_template)

    mask = None
    if game_phase == 2:
        roi_gray_np = cv2.bitwise_not(roi_gray_np)

    _, mask = cv2.threshold(roi_gray_np, 200, 240, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 10), np.uint8)
    dilation = cv2.dilate(mask, kernel)

    _, contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    to_return = list()
    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)
        if h < 8:
            continue
        rectangle = Rectangle(x, y, x + w, y + h)
        to_return.append(rectangle)

    return to_return, enemy_segment_template.shape()
