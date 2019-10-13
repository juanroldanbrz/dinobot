from typing import List

import numpy as np
from scipy.spatial import distance

from model.learning_model import LearningModel
from model.rectangle import Rectangle


def test_model(model: LearningModel, enemies_rectangle: List[Rectangle], img_shape):
    if len(enemies_rectangle) == 0:
        return 'continue'

    enemies_rectangle.sort(key=lambda x: x.x1)
    enemy_rectangle = enemies_rectangle[0]
    p_top_left = (enemy_rectangle.x1, enemy_rectangle.y1)
    p_top_right = (enemy_rectangle.x2, enemy_rectangle.y1)

    p_bot_left_img = (0, img_shape[0])
    total_distance = distance.euclidean((0, 0), (img_shape[0], img_shape[1]))

    x1 = distance.euclidean(p_bot_left_img, p_top_left) / total_distance
    x2 = distance.euclidean(p_bot_left_img, p_top_right) / total_distance

    x_vector = np.array(x1, x2).reshape(1, 2)
    return model.decide(x_vector)
