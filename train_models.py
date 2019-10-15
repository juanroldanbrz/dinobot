import numpy as np
import cv2
import pyscreenshot as ImageGrab

from actor.dino_detector_actor import DinoDetectorActor
from actor.enemy_detector import EnemyDetectorActor
from actor.game_simulation import GameSimulation
from actor.game_status import GameStatusActor
from service import model_manager
from model.message import Message
from template.template import screen_template, enemy_segment_template
from utils import utils

game_status_actor = GameStatusActor()
enemy_detector = EnemyDetectorActor()
dino_detector = DinoDetectorActor()

model = model_manager.get_next_model('test_model.csv')
game_actor = GameSimulation(model, game_status_actor)

while True:
    img = ImageGrab.grab(bbox=screen_template.to_tuple())
    img_np = np.array(img)
    img_np = utils.resize_to_rectangle(img_np, screen_template)

    # Detect enemies
    msg = Message('detect_enemy', utils.to_gray(img_np))
    enemy_detector.tell(msg)

    # Detect game status
    msg = Message('process_status', utils.to_gray(img_np))
    game_status_actor.tell(msg)


    # cv2.putText(img_np, status, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

    enemies = enemy_detector.ask('enemies')
    for enemy in enemies:
        utils.display_rectangle(img_np, enemy)

    if dino:
        utils.display_rectangle(img_np, dino, 50)

    cv2.imshow("frame", img_np)
    key = cv2.waitKey(1)
    if key == 27:
        break
