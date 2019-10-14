import unittest

import cv2
import numpy as np

from actor.decision_actor import DecisionActor
from actor.dino_detector_actor import DinoDetectorActor
from actor.enemy_detector import EnemyDetectorActor
from model.message import Message
from template.template import enemy_segment_template
from utils import utils

decision_actor = DecisionActor()
enemy_detector = EnemyDetectorActor()
dino_detector = DinoDetectorActor()


class DecisionActorTest(unittest.TestCase):
    def test_decide_actor(self):
        img = cv2.imread('../snapshots/enemy/enemies_3.png')
        img_np = np.array(img)

        # Detect enemies
        msg = Message('detect_enemy', utils.to_gray(img_np))
        enemy_detector.tell(msg)

        msg = Message('detect_dino', utils.to_gray(img_np))
        dino_detector.tell(msg)
        dino = dino_detector.ask('dino')

        enemies = enemy_detector.ask('enemies')

        msg_content = {
            'img_np': utils.crop_image(img_np, enemy_segment_template),
            'enemies': list(map(lambda x: x.relativize_to(enemy_segment_template), enemies))
        }
        msg = Message('decide', msg_content)
        decision_actor.tell(msg)
