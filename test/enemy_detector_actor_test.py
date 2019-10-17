import unittest

import cv2
import numpy as np

from actor import enemy_detector
from utils import utils


class EnemyDetectorActorTest(unittest.TestCase):
    def test_detect_enemy_2(self):
        img = cv2.imread('../snapshots/enemy/enemies_2_full.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        enemy_detector.find_enemies(gray)

    def test_detect_no_enemy(self):
        img = cv2.imread('../snapshots/enemy/no_enemies.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        enemies, _ = enemy_detector.find_enemies(gray)
        self.assertEqual(0, len(enemies))

    def test_detect_no_enemy_2(self):
        img = cv2.imread('../snapshots/enemy/no_enemy_1.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        enemies, _ = enemy_detector.find_enemies(gray)
        self.assertEqual(0, len(enemies))

    def test_detect_enemy_5(self):
        img = cv2.imread('../snapshots/enemy/enemies_5.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        enemies, _ = enemy_detector.find_enemies(gray)
        self.assertEqual(1, len(enemies))
