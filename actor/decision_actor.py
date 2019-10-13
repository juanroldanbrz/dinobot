from typing import List

import numpy as np

from scipy.spatial import distance

from model.actor import Actor
from model.rectangle import Rectangle


def segmentize(x_segments, y_segments, img_np, enemies: List[Rectangle]):
    result = np.zeros((y_segments, x_segments))

    segment_x_size = img_np.shape[1] / x_segments
    segment_y_size = img_np.shape[0] / y_segments

    for rectangle in enemies:
        x1 = int(rectangle.x1 / segment_x_size)
        x2 = int(rectangle.x2 / segment_x_size)

        y1 = int(rectangle.y1 / segment_y_size)
        y2 = int(rectangle.y2 / segment_y_size)

        result[y1:y2, x1:x2] = 1



class DecisionActor(Actor):

    def __init__(self):
        self.decision = 'stay'
        self.train = True

    def tell(self, message):
        if message.message_type == 'decide':
            if self.train:
                train(message.content['img_np'].shape, message.content['enemies'])

    def ask(self, message_type: str):
        if message_type == 'decide':
            if self.train:
                train(message.content['img_np'].shape, message.content['enemies'])

            return self.decision
