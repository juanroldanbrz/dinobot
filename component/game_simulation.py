import time

from component import game_status, enemy_detector
from model.learning_model import LearningModel
from service import learning_model_service


class GameSimulation:
    def __init__(self, model: LearningModel):
        self.model = model
        self.status = 'inactive'
        self.start_time = None
        self.end_time = None

    def start(self):
        self.status = 'playing'
        self.start_time = time.time()

    def terminate(self):
        self.status = 'terminated'
        self.end_time = time.time()

    def play(self, full_gray_np):
        if self.status == 'game_over':
            raise AssertionError('Cannot play a terminated game')

        status = game_status.get_game_status(full_gray_np)
        if status == 'game_over':
            self.terminate()
            return 'terminate'

        enemies, roi_shape = enemy_detector.find_enemies(full_gray_np)
        if len(enemies) == 0:
            return 'continue'

        result = learning_model_service.test_model(self.model, enemies, roi_shape)

        if result > 0.5:
            return 'jump'
        elif result < -0.5:
            return 'down'
        else:
            return 'continue'
