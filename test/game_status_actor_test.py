import unittest

import cv2
import numpy as np

from actor.game_status_actor import GameStatusActor
from model.message import Message
from utils import utils

game_status_actor = GameStatusActor()


class GameStatusActorTest(unittest.TestCase):
    def test_image_should_be_gray(self):
        img = cv2.imread('../snapshots/game_over.png')
        img_np = np.array(img)
        msg = Message('process_status', img_np)
        with self.assertRaises(AssertionError) as context:
            game_status_actor.tell(msg)
            self.assertTrue('This is broken' in context.exception)

    def test_game_over_status(self):
        img = cv2.imread('../snapshots/game_over.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        msg = Message('process_status', gray)
        game_status_actor.tell(msg)
        self.assertEqual('game_over', game_status_actor.ask('game_status'))

    def test_playing_status(self):
        img = cv2.imread('../snapshots/game_playing.png')
        img_np = np.array(img)
        gray = utils.to_gray(img_np)
        msg = Message('process_status', gray)
        game_status_actor.tell(msg)
        self.assertEqual('playing', game_status_actor.ask('game_status'))

    def test_invalid_size_image(self):
        img = cv2.imread('../snapshots/game_over_invalid_size.png')
        img_np = np.array(img)
        msg = Message('process_status', img_np)

        with self.assertRaises(AssertionError) as context:
            game_status_actor.tell(msg)
            self.assertTrue('This is broken' in context.exception)
