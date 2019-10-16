import random
from typing import List

import numpy as np
from pymongo import MongoClient
from scipy.spatial import distance

from model.learning_model import LearningModel
from model.rectangle import Rectangle
from service.mapper import from_model_to_bxon, from_bson_to_model

client = MongoClient('localhost', 27017, connecttimeoutms=1000)
db = client.dino_game_db
collection = db.learning_model


def insert_one(model: LearningModel):
    to_store = from_model_to_bxon(model)
    collection.insert_one(to_store)


def insert_many(models: List[LearningModel]):
    to_store = list(map(lambda x: from_model_to_bxon(x), models))
    collection.insert_many(to_store)


def find_one(processed=False) -> LearningModel:
    bson = collection.find_one({"processed": processed})
    return from_bson_to_model(bson)


def update_score(model_id: str, score):
    collection.update({'model_id': model_id}, {"$set": {"score": score, "processed": True}}, upsert=False)


def generate_models(num_models: int) -> List[LearningModel]:
    to_return = list()
    for i in range(0, num_models):
        w_vector = list(i for i in (np.random.rand(1, 2) + 1).flatten())
        d = random.rand(-5, 5)
        to_return.append(LearningModel(w_vector, d))
    return to_return


def test_model(model: LearningModel, enemies_rectangle: List[Rectangle], img_shape):
    if len(enemies_rectangle) == \
            0:
        return 0

    enemies_rectangle.sort(key=lambda x: x.x1)
    enemy_rectangle = enemies_rectangle[0]
    p_top_left = (enemy_rectangle.x1, enemy_rectangle.y1)
    p_top_right = (enemy_rectangle.x2, enemy_rectangle.y1)

    p_bot_left_img = (0, img_shape[0])
    total_distance = distance.euclidean((0, 0), (img_shape[0], img_shape[1]))

    x1 = distance.euclidean(p_bot_left_img, p_top_left) / total_distance
    x2 = distance.euclidean(p_bot_left_img, p_top_right) / total_distance

    x_vector = np.array([[x1, x2]])
    to_return = model.apply(x_vector)
    print(f'Model {model.model_id} calculated {to_return}')
    return to_return
