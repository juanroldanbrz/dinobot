from time import sleep

import cv2
import numpy as np
import pyscreenshot as ImageGrab

from actor import enemy_detector, dino_detector_actor, game_status
from actor.game_runner import GameRunner
from service import learning_model_service
from template.template import screen_template, enemy_segment_template
from utils import utils

model = learning_model_service.find_one()
game_runner = GameRunner(model)
print(f''
      f'Loading model: {model.model_id}')

print('Starting in 5...')

img = ImageGrab.grab(bbox=screen_template.to_tuple(), childprocess=False)
img_np = np.array(img)
cv2.imshow('game', img_np)

print('check')
game_runner.start()

while True:
    print('loop')
    img = ImageGrab.grab(bbox=screen_template.to_tuple(), childprocess=False)
    img_np = np.array(img)
    img_np = utils.resize_to_rectangle(img_np, screen_template)
    full_gray_np = utils.to_gray(img_np)

    # Detect enemies
    enemies, _ = enemy_detector.find_enemies(full_gray_np)
    status = f'{game_runner.status} - w:{model.w_vector}, d:{model.d}'

    game_runner.play(enemies, enemy_segment_template.shape())
    game_status_str = game_status.get_game_status(utils.to_gray(img_np))

    cv2.putText(full_gray_np, status, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

    for enemy in enemies:
        utils.display_rectangle(full_gray_np, enemy.relativize_from(enemy_segment_template))

    cv2.imshow('game', full_gray_np)
    if game_status_str == 'game_over':
        game_runner.terminate()
        sleep(5)
        print('finished')
    #
    # game_runner.play(full_gray_np)
    #
    # msg = f'id: {model.model_id}, w: {model.w_vector}, d: {model.d}'
    #
    # cv2.putText(img_np, msg, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    #
    # if game_runner.terminated:
    #     print(f'Game finished. score: {game_runner.get_score()}')
    #     model = learning_model_service.find_one()
    #     game_simulation = GameSimulation(model)
    #     game_runner = GameRunner(game_simulation)
    #     print(f'Loading model: {model}')
    #     game_runner.start()
    #     sleep(5)
