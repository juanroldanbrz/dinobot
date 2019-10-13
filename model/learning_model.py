import uuid

import numpy as np


def from_csv_entry(csv_entry):
    id, w_serialized, d_str = csv_entry.split(',')
    d = int(d_str)
    w_str_list = w_serialized.replace('[', '').replace(']', '').split(' ')
    w_str_list = list(filter(lambda x: x != '', w_str_list))
    w = list(map(lambda x: float(x), w_str_list))
    w_np = np.array(w).reshape(1, 2)
    return LearningModel(w_np, d, id)


class LearningModel:
    def __init__(self, w_vector, d: float, id=None):
        if id:
            self.id = id
        else:
            self.id = str(uuid.uuid4())
        self.w_vector = w_vector
        self.d = d

    def decide(self, x_vector):
        product = x_vector * self.w_vector + self.d
        average = np.sum(product, axis=1) / x_vector.shape[1]
        tan = np.tan(average).ravel()

        if tan > 0.5:
            return tan, 'jump'
        elif tan < -0.5:
            return tan, 'down'
        else:
            return tan, 'continue'

    def to_csv_entry(self):
        return f'{self.id},{self.w_vector.ravel()},{self.d}\n'
