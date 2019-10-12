import numpy as np
import cv2
import pyscreenshot as ImageGrab

from actor.enemy_detector_actor import EnemyDetectorActor
from actor.game_status_actor import GameStatusActor
from model.message import Message
from template.template import screen_template, enemy_segment_template
from utils import utils

game_status_actor = GameStatusActor()
enemy_detector = EnemyDetectorActor()
while True:
    i = 2
    img = ImageGrab.grab(bbox=screen_template.to_tuple())
    img_np = np.array(img)
    img_np = utils.resize_to_rectangle(img_np, screen_template)
    gray = utils.to_gray(img_np)
    msg = Message('detect_enemy', gray)
    enemy_detector.tell(msg)
    enemies = enemy_detector.ask('enemies')
    for enemy in enemies:
        utils.display_rectangle(img_np, enemy)

    cv2.imshow("frame", img_np)
    key = cv2.waitKey(1)
    if key == 27:
        break
