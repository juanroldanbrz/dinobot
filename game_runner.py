from time import sleep

import cv2
import numpy as np
from PIL import ImageGrab

from actor.game_runner import GameRunner
from actor.game_simulation import GameSimulation
import pyautogui

from service import learning_model_service
from template.template import screen_template
from utils import utils

model = learning_model_service.find_one()
game_simulation = GameSimulation(model)
game_runner = GameRunner(game_simulation)
print(f'Loading model: {model.model_id}')
game_runner.start()

while True:
    img = ImageGrab.grab(bbox=screen_template.to_tuple())
    img_np = np.array(img)
    cv2.imshow("frame", img_np)
    cv2.waitKey(0)
    img_np = utils.resize_to_rectangle(img_np, screen_template)
    full_gray_np = utils.to_gray(img_np)


    game_runner.play(full_gray_np)


    msg = f'id: {model.model_id}, w: {model.w_vector}, d: {model.d}'

    cv2.putText(img_np, msg, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

    if game_runner.terminated:
        print(f'Game finished. score: {game_runner.get_score()}')
        model = learning_model_service.find_one()
        game_simulation = GameSimulation(model)
        game_runner = GameRunner(game_simulation)
        print(f'Loading model: {model}')
        game_runner.start()
        sleep(5)

    key = cv2.waitKey(1)
    if key == 27:
        break
