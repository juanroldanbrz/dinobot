import time

import pyautogui

from actor.game_simulation import GameSimulation
from model.learning_model import LearningModel
from model.rectangle import Rectangle
from service import learning_model_service

JUMP_KEY = 'space'
DOWN_KEY = 'down'
INIT_KEY = 'enter'


class GameRunner:
    def __init__(self, model: LearningModel):
        self.model = model
        self.status = 'not_processed'

        self.start_time = None
        self.end_time = None

    def start(self):
        print('Starting game in 3...')
        time.sleep(4)
        self.start_time = time.time()
        pyautogui.press(INIT_KEY)
        self.status = 'playing'

    def play(self, enemies: [Rectangle], roi_shape):
        if len(enemies) == 0:
            return 0
        result = learning_model_service.test_model(self.model, enemies, roi_shape)
        if result >= 0.5:
            pyautogui.press(JUMP_KEY)
            # print('jumping')
        elif result <= - 0.5:
            pyautogui.press(DOWN_KEY)
            # print('down')

    def terminate(self):
        self.end_time = time.time()
        self.status = 'terminated'
        print('Terminated')

    def get_score(self):
        return self.end_time - self.start_time
