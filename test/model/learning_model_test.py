import random
import unittest

import numpy as np

from model.learning_model import LearningModel


class LearningModelTest(unittest.TestCase):
    def test_object_instance(self):
        for i in range(1, 100):
            w = np.random.rand(1, 2) + 1
            d = random.randint(-5, 5)
            learning_model = LearningModel(w, d)
            x = np.random.rand(1, 2)
            value = learning_model.decide(x)
            print(f'value {value}')
