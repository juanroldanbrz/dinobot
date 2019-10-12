from typing import List

import cv2
import numpy as np

from model.actor import Actor
from model.rectangle import Rectangle
from template.template import screen_template, enemy_segment_template
from utils import utils
from utils.assertions import assert_rectangle_shape, assert_gray_img


def find_enemies(full_gray_np) -> List[Rectangle]:
    # Assert
    assert_rectangle_shape(full_gray_np, screen_template, f'Shape of the image should be {screen_template.shape()}')
    assert_gray_img(full_gray_np)

    roi_gray_np = utils.crop_image(full_gray_np, enemy_segment_template)
    cv2.imshow('roi_gray_np', roi_gray_np)

    _, mask = cv2.threshold(roi_gray_np, 200, 240, cv2.THRESH_BINARY_INV)

    kernel = np.ones((4, 4), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(opening, kernel)

    _, contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    to_return = list()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rectangle = Rectangle(x, y, x + w, y + h)
        rectangle = rectangle.relativize_from(enemy_segment_template)
        to_return.append(rectangle)

    return to_return


class EnemyDetectorActor(Actor):
    def __init__(self):
        self.enemies = []

    def tell(self, message):
        if message.message_type == 'detect_enemy':
            self.enemies = find_enemies(message.content)

    def ask(self, message_type: str):
        if message_type == 'enemies':
            return self.enemies
