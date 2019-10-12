import unittest

import cv2
import numpy as np

from actor.enemy_detector_actor import EnemyDetectorActor
from actor.game_status_actor import GameStatusActor
from model.message import Message
from utils import utils

enemy_detector = EnemyDetectorActor()


class EnemyDetectorActorTest(unittest.TestCase):
    def test_detect_enemy_1(self):
        img = cv2.imread('../snapshots/enemy/enemies_1_full.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        msg = Message('detect_enemy', gray)
        enemy_detector.tell(msg)
        enemies = enemy_detector.ask('enemies')
        self.assertEqual(2, len(enemies))

    def test_detect_enemy_2(self):
        img = cv2.imread('../snapshots/enemy/enemies_3.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        msg = Message('detect_enemy', gray)
        enemy_detector.tell(msg)
        enemies = enemy_detector.ask('enemies')
        self.assertEqual(1, len(enemies))

    def test_detect_enemy_3(self):
        img = cv2.imread('../snapshots/enemy/enemies_4.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        msg = Message('detect_enemy', gray)
        enemy_detector.tell(msg)
        enemies = enemy_detector.ask('enemies')
        self.assertEqual(2, len(enemies))
