import uuid

import numpy as np


class LearningModel:
    def __init__(self, w_vector: [float], d: float, model_id=None):
        self.model_id = model_id if model_id else str(uuid.uuid4())
        self.w_vector = w_vector
        self.d = d

        self.score = None
        self.processed = False

        self.generation = None

    def apply(self, x_vector: [float]):
        x = np.array(x_vector).reshape(1, 3)
        w = np.array([self.w_vector])

        product = x * w + self.d
        average = np.sum(product, axis=1) / x_vector.shape[1]
        return np.tan(average).ravel()

    def __hash__(self):
        return hash((str(self.w_vector), self.d))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.w_vector == other.w_vector and self.d == other.d
