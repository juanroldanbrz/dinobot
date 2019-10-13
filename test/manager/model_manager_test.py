import unittest

from learning_model.model_manager import ModelManager

model_manager = ModelManager()

model_path = 'test.model'


class ModelManagerTest(unittest.TestCase):
    def test_create_models(self):
        model_manager.create_model(model_path, 100)

    def test_load_models(self):
        models = model_manager.load_models(model_path)
        self.assertEqual(100, len(models))

    def test_next_model(self):
        model = model_manager.get_next_model(model_path)
        self.assertIsNotNone(model)

    def test_add_to_processed(self):
        model = model_manager.get_next_model(model_path)
        model_manager.add_to_processed(model_path, model.id, 0.123)
        self.assertNotEqual(model.id, model_manager.get_next_model(model_path))
