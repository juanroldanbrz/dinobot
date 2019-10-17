import unittest

import cv2
import numpy as np

from actor import enemy_detector
from utils import utils


class EnemyDetectorActorTest(unittest.TestCase):
    def test_detect_enemy_3(self):
        img = cv2.imread('../snapshots/enemy/no_enemies.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        enemy_detector.find_enemies(gray)

