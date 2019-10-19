import copy
import random
from typing import List

import numpy as np

from model.learning_model import LearningModel
from service import learning_model_service


def reproduce_generation(gen_number: int):
    next_gen = gen_number + 1
    to_return = []
    ids = {}

    last_generation = learning_model_service.find_all(processed=True, generation=gen_number)
    last_generation.sort(key=lambda x: x.score, reverse=True)
    best_models = last_generation[:10]
    to_return += copy.deepcopy(best_models)
    # to_return += _reproduce(best_models)
    to_return += _mutate(best_models, gen=gen_number)

    for model in to_return:
        model.generation = next_gen
        model.processed = False
        model.score = None
        model_id = model.model_id
        if model_id in ids:
            ids[model_id] = ids[model_id] + 1
        else:
            ids[model_id] = 1

    learning_model_service.insert_many(to_return)


def _mutate(models: List[LearningModel], gen):
    to_return = []

    for model in models:
        if gen <= 3:
            big_variation = _mutate_values(model, -0.8, 0.8)
            half = _mutate_values(model, -0.5, 0.5)
            quarter = _mutate_values(model, -0.25, 0.25)
            low = _mutate_values(model, -0.10, 0.10)
            to_return.append(half)
            to_return.append(quarter)
            to_return.append(low)
            to_return.append(big_variation)
        else:
            for i in range(1, 4):
                to_return.append(_mutate_values(model, -1, 1))
                to_return.append(_mutate_values(model, -0.25, 0.25))
                to_return.append(_mutate_values(model, -0.10, 0.10))

            to_return.append(_mutate_values(model, -3.25, 3.25))

    return to_return


def _mutate_values(model, from_val, to_val):
    d = model.d
    w_original = np.array(model.w_vector)
    half_vector = np.random.uniform(from_val, to_val, (1, 3))
    w_mut = copy.deepcopy(w_original) + (copy.deepcopy(w_original) * half_vector)
    d_half = d + (d * random.uniform(from_val, to_val))
    return create_model(w_mut, d_half)


# len input * 2
def _reproduce(models: List[LearningModel]):
    fathers = copy.deepcopy(models)
    mothers = copy.deepcopy(models)

    to_return = []

    for i in range(0, 3):
        random.shuffle(fathers)

        for father in fathers:
            random.shuffle(mothers)
            mother = mothers[0]
            while father.model_id == mother.model_id:
                random.shuffle(mothers)
                mother = mothers[0]
            to_return.append(_make_child(father, mother))

    limit = 2 * len(models)
    set_return = set(to_return)
    if len(set_return) < limit:
        return list(set_return)
    else:
        return list(set_return)[:limit]


def create_generation_report(gen: int):
    models = learning_model_service.find_all(processed=True, generation=gen)
    if len(models) == 0:
        raise AssertionError(f'no models for generation {gen}')
    models.sort(key=lambda x: x.score, reverse=True)
    best_model = models[0]
    models_top_5_average = sum(list(map(lambda x: x.score, models[:5]))) / 5.0
    models_top_5_score = list(map(lambda x: x.score, models[:5]))

    f = open(f'generation_{gen}.txt\n', 'w')
    f.write(f'Generation: {gen}\n')
    f.write(f'Best model score: {best_model.score}\n')
    f.write(f'Top 5 models score: {str(models_top_5_score)}\n')
    f.write(f'Top 5 average: {str(models_top_5_average)}\n')
    f.close()


def create_model(w_array, d):
    w_array_mut = [i for i in w_array.ravel()]
    model = LearningModel(w_array_mut, d)
    model.processed = False
    return model


def _make_child(father, mother):
    w_father = np.array(father.w_vector)
    w_mother = np.array(mother.w_vector)

    d_father = father.d
    d_mother = mother.d

    w_child = (w_father + w_mother * 1.0) / 2
    d_child = (d_father + d_mother * 1.0) / 2

    return create_model(w_child, d_child)
