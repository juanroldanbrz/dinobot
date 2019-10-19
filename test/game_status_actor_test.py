import unittest

import cv2
import numpy as np

from component import game_status
from utils import utils


class GameStatusActorTest(unittest.TestCase):
    def test_game_phase_1(self):
        img = cv2.imread('../snapshots/game_playing.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        phase = game_status.get_phase(gray)
        self.assertEqual(1, phase)

    def test_game_phase_2_1(self):
        img = cv2.imread('../snapshots/black_background_template.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        phase = game_status.get_phase(gray)
        self.assertEqual(2, phase)

    def test_game_phase_1_2(self):
        img = cv2.imread('../snapshots/still_phase_1.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        phase = game_status.get_phase(gray)
        self.assertEqual(1, phase)

    def test_game_phase_2_2(self):
        img = cv2.imread('../snapshots/phase2/enemy/enemy1.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        phase = game_status.get_phase(gray)
        self.assertEqual(1, phase)


