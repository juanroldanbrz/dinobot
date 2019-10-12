import cv2

from model.actor import Actor
from template.template import screen_template, enemy_segment_template
from utils import utils
from utils.assertions import assert_rectangle_shape, assert_gray_img


def detect_enemies(gray_np) -> str:
    # Assert
    assert_rectangle_shape(gray_np, screen_template, f'Shape of the image should be {screen_template.shape()}')
    assert_gray_img(gray_np)

    utils.display_rectangle(gray_np, enemy_segment_template)
    cv2.imshow('enemy_segment', gray_np)
    cv2.waitKey(0)
    # enemy_segment_img_gray = utils.crop_image(gray_np, enemy_segment_template)
    return 'ok'


class EnemyDetectorActor(Actor):
    def __init__(self):
        self.detected_enemies = []

    def tell(self, message):
        if message.message_type == 'detect_enemy':
            self.detected_enemies = detect_enemies(message.content)

    def ask(self, message_type: str):
        if message_type == 'enemies':
            return self.detected_enemies
