import cv2
import numpy as np
import pyscreenshot as ImageGrab

from actor import enemy_detector, dino_detector_actor, game_status
from template.template import screen_template, enemy_segment_template
from utils import utils

while True:

    print('loop')

    img = ImageGrab.grab(bbox=screen_template.to_tuple())
    img_np = np.array(img)
    img_np = utils.resize_to_rectangle(img_np, screen_template)


    # Detect enemies
    enemies, _ = enemy_detector.find_enemies(utils.to_gray(img_np))
    dino = dino_detector_actor.find_dino(utils.to_gray(img_np))
    status = game_status.get_game_status(utils.to_gray(img_np))

    cv2.putText(img_np, status, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

    for enemy in enemies:
        utils.display_rectangle(img_np, enemy.relativize_from(enemy_segment_template))

    if dino:
        utils.display_rectangle(img_np, dino, 50)

    cv2.imshow("frame", img_np)
    key = cv2.waitKey(1)
    if key == 27:
        break
