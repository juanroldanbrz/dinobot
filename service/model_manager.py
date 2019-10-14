import random
from typing import List

import numpy as np

from model import learning_model
from model.learning_model import LearningModel


def save_model(serialized_model, models_path: str):
    with open(models_path, "a") as models:
        models.write(serialized_model)


def create_model(models_path, iterations=1):
    for i in range(0, iterations):
        w = np.random.rand(1, 2) + 1
        d = random.randint(-5, 5)
        learning_model = LearningModel(w, d)
        save_model(learning_model.to_csv_entry(), models_path)


def load_models(models_path: str) -> List[LearningModel]:
    with open(models_path, "r") as models:
        lines = models.readlines()
        models = list(map(lambda x: learning_model.from_csv_entry(x), lines))
        return models


def get_next_model(models_path: str) -> LearningModel:
    models = load_models(models_path)

    solutions_file = f'{models_path}.result'
    open(solutions_file, 'a').close()

    ids = list()
    with open(solutions_file, "r") as processed:
        lines = processed.readlines()
        if lines:
            ids = list(map(lambda x: x.split(',')[0], lines))

    for model in models:
        if model.id not in ids:
            return model


def add_to_processed(models_path, model_id, elapsed_ms):
    solutions_file = f'{models_path}.result'
    with open(solutions_file, "a") as processed:
        processed.write(f'{model_id},{elapsed_ms}\n')
