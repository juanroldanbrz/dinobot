import unittest

import cv2
import numpy as np

from actor import enemy_detector, game_status
from utils import utils


class EnemyDetectorActorTest(unittest.TestCase):
    ## Complicated
    def test_detect_enemy_complicated_cactus_3(self):
        img = cv2.imread(f'../snapshots/complicated/cactus3.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        phase = game_status.get_phase(gray)
        enemies, _ = enemy_detector.find_enemies(gray, phase)
        self.assertEqual(1, len(enemies))

    def test_detect_enemy_complicated_1(self):
        enemy_files = ['bird.png', 'bird2.png', 'bird3.png', 'bird4.png', 'cactus.png', 'cactus2.png']

        for enemy_file in enemy_files:
            img = cv2.imread(f'../snapshots/complicated/{enemy_file}')
            img_np = np.array(img)
            gray = utils.to_gray(img_np)
            phase = game_status.get_phase(gray)
            enemies, _ = enemy_detector.find_enemies(gray, phase)
            self.assertEqual(1, len(enemies))

    ###
    def test_detect_enemy_phase_2(self):
        img = cv2.imread('../snapshots/phase2/enemy/enemy1.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        enemies, _ = enemy_detector.find_enemies(gray, 2)
        self.assertEqual(1, len(enemies))

    def test_detect_enemy_phase_2_3(self):
        img = cv2.imread('../snapshots/phase2/enemy/enemy_3.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        enemies, _ = enemy_detector.find_enemies(gray, 2)
        self.assertEqual(1, len(enemies))

    def test_detect_enemy_phase_2_2(self):
        img = cv2.imread('../snapshots/phase2/enemy/enemy2.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        enemies, _ = enemy_detector.find_enemies(gray, 2)
        self.assertEqual(1, len(enemies))

    def test_detect_enemy_phase_2_1(self):
        img = cv2.imread('../snapshots/phase2/enemy/enemy1.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        enemies, _ = enemy_detector.find_enemies(gray, 2)
        self.assertEqual(1, len(enemies))

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
