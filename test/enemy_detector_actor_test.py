import unittest

import cv2
import numpy as np

from actor.enemy_detector_actor import EnemyDetectorActor
from actor.game_status_actor import GameStatusActor
from model.message import Message
from utils import utils

enemy_detector = EnemyDetectorActor()


class EnemyDetectorActorTest(unittest.TestCase):
    def test_detect_enemy(self):
        img = cv2.imread('../snapshots/enemy/enemies_1_full.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        msg = Message('detect_enemy', gray)
        enemy_detector.tell(msg)

    def test_invalid_size_image(self):
        img = cv2.imread('../snapshots/game_over_invalid_size.png')
        img_np = np.array(img)
        msg = Message('process_status', img_np)

        with self.assertRaises(AssertionError) as context:
            game_status_actor.tell(msg)
            self.assertTrue('This is broken' in context.exception)
