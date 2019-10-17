import math
import random
from typing import List

import numpy as np
from pymongo import MongoClient
from scipy.spatial import distance

from model.learning_model import LearningModel
from model.rectangle import Rectangle
from service.mapper import from_model_to_bson, from_bson_to_model

client = MongoClient('localhost', 27017, connecttimeoutms=1000)
db = client.dino_game_db
collection = db.model_generations


def insert_one(model: LearningModel):
    to_store = from_model_to_bson(model)
    collection.insert_one(to_store)


def insert_many(models: List[LearningModel]):
    to_store = list(map(lambda x: from_model_to_bson(x), models))
    collection.insert_many(to_store)


def find_one(processed=False, generation=1) -> LearningModel:
    bson = collection.find_one({"processed": processed, 'generation': generation})
    return from_bson_to_model(bson)


def find_all(processed=False, generation=1) -> List[LearningModel]:
    bson = collection.find({"processed": processed, 'generation': generation})
    return list(map(lambda x: from_bson_to_model(x), bson))


def update_score(model_id: str, score):
    collection.update({'model_id': model_id}, {"$set": {"score": score, "processed": True}}, upsert=False)


def generate_models(num_models: int, specie='a', generation=1) -> List[LearningModel]:
    to_return = list()
    size = (1, 3)

    for i in range(0, num_models):
        if specie == 'a':
            w_vector = list(i for i in (np.random.uniform(-2, 2, size)).flatten())
            d = random.randrange(-3, 3)
            model = LearningModel(w_vector, d)
            model.generation = generation
            to_return.append(model)
        elif specie == 'b':
            w_vector = list(i for i in (np.random.uniform(-5, 5, size)).flatten())
            d = random.randrange(-1, 1)
            model = LearningModel(w_vector, d)
            model.generation = generation
            to_return.append(model)
        elif specie == 'c_rare':
            w_vector = list(i for i in (np.random.uniform(-10, 10, size)).flatten())
            d = random.randrange(-10, 10)
            model = LearningModel(w_vector, d)
            model.generation = generation
            to_return.append(model)

    return to_return


def get_angle(p1, p2):
    deltax = p2[0] - p1[0] * 1.0
    deltay = p1[1] - p2[1] * 1.0

    angle_rad = math.atan2(deltay, deltax)
    return angle_rad * 180.0 / math.pi


def test_model(model: LearningModel, enemies_rectangle: List[Rectangle], img_shape):
    if len(enemies_rectangle) == \
            0:
        return 0

    p_bot_left_img = (0, img_shape[0])

    enemies_rectangle.sort(key=lambda x: x.x1)
    enemy_rectangle = enemies_rectangle[0]
    p_top_left = (enemy_rectangle.x1, enemy_rectangle.y1)
    p_top_right = (enemy_rectangle.x2, enemy_rectangle.y1)
    p_bot_left = (enemy_rectangle.x1, enemy_rectangle.y2)

    total_distance = distance.euclidean((0, 0), (img_shape[0], img_shape[1]))

    x1 = distance.euclidean(p_bot_left_img, p_top_left) / total_distance
    x2 = distance.euclidean(p_bot_left_img, p_top_right) / total_distance
    x3 = (get_angle(p_bot_left_img, p_bot_left) / 90) + 1

    x_vector = np.array([[x1, x2, x3]])
    to_return = model.apply(x_vector)
    print(f'Model {model.model_id} calculated {to_return}')
    return to_return
