from time import sleep

import cv2
import numpy as np
import pyscreenshot as ImageGrab

from actor import enemy_detector, dino_detector_actor, game_status
from actor.game_runner import GameRunner
from service import learning_model_service
from service.learning_model_service import update_score
from template.template import screen_template, enemy_segment_template
from utils import utils

d = np.random.random(1)
model = learning_model_service.find_one()
game_runner = GameRunner(model)
print(f'Loading model: {model.model_id}')
game_runner.start()

while True:
    img = ImageGrab.grab(bbox=screen_template.to_tuple(), childprocess=False)
    img_np = np.array(img)
    img_np = utils.resize_to_rectangle(img_np, screen_template)
    full_gray_np = utils.to_gray(img_np)

    # Detect enemies
    enemies, _ = enemy_detector.find_enemies(utils.to_gray(img_np))
    status = f'{game_runner.status} - w:{model.w_vector}, d:{model.d}, specie:{model.specie}'

    game_runner.play(enemies, enemy_segment_template.shape())
    game_status_str = game_status.get_game_status(utils.to_gray(img_np))

    cv2.putText(full_gray_np, status, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0)

    for enemy in enemies:
        utils.display_rectangle(full_gray_np, enemy.relativize_from(enemy_segment_template))

    cv2.imshow('game2', full_gray_np)
    cv2.waitKey(1)

    if game_status_str == 'game_over':
        game_runner.terminate()
        print(f'elapsed: {game_runner.get_score()}')
        update_score(model.model_id, game_runner.get_score())

        model = learning_model_service.find_one()
        game_runner = GameRunner(model)
        print(f'Loading model: {model.model_id}')
        game_runner.start()



