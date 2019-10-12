import unittest

import cv2
import numpy as np

from actor.dino_detector_actor import DinoDetectorActor
from actor.enemy_detector_actor import EnemyDetectorActor
from actor.game_status_actor import GameStatusActor
from model.message import Message
from utils import utils

dino_detector = DinoDetectorActor()


class EnemyDetectorActorTest(unittest.TestCase):
    def test_detect_dino_1(self):
        img = cv2.imread('../snapshots/enemy/enemies_3.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        msg = Message('detect_dino', gray)
        dino_detector.tell(msg)
        dino = dino_detector.ask('dino')
        self.assertAlmostEqual(2100, dino.area(), delta=12)