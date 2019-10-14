import unittest

from model.learning_model import LearningModel
from service import learning_model_service


class ModelManagerTest(unittest.TestCase):
    def test_generate_models(self):
        learning_model_service.generate_models(100)

    def test_insert_many(self):
        learning_model_service.generate_models(100)

    def test_find_one(self):
        model = learning_model_service.find_one()
        self.assertIsInstance(model, LearningModel)