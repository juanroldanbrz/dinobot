import datetime
import time

from actor.game_status_actor import GameStatusActor
from learning_model.test_model import test_model
from model.learning_model import LearningModel


class GameActor:
    def __init__(self, model: LearningModel, game_status_actor: GameStatusActor):
        self.initialize(model)

        self.model = model
        self.status = 'inactive'
        self.action = None
        self.start_time = None
        self.end_time = None

        self.game_status_actor = game_status_actor

    def initialize(self, model):
        self.model = model
        self.start_time = None
        self.end_time = None
        self.action = 'continue'
        self.status = 'inactive'

    def tell(self, message):
        if message.message_type == 'play':
            if self.status == 'inactive':
                self.action = 'key_enter'
                self.status = 'playing'
                self.start_time = time.time()

            elif self.action == 'playing':
                if self.game_status_actor.ask('game_status') == 'game_over':
                    self.status = 'terminated'
                    self.end_time = time.time()
                else:
                    shape, enemies = message.content['img_np'].shape, message.content['enemies']
                    self.action = test_model(self.model, enemies, shape)
        elif message.message_type == 'reload':
            self.initialize(message.content)

    def ask(self, message_type: str):
        if message_type == 'action':
            return self.action

        if message_type == 'status':
            return self.status

        if message_type == 'elapsed':
            return self.end_time - self.start_time
