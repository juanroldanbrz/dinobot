import numpy as np
import cv2
import pyscreenshot as ImageGrab

from actor.game_status_actor import GameStatusActor
from model.message import Message
from template.template import screen_template, enemy_segment_template
from utils import utils

game_status_actor = GameStatusActor()
while True:
    i = 2
    img = ImageGrab.grab(bbox=screen_template.to_tuple())  # x, y, x2, y2
    img_np = np.array(img)
    gray = utils.to_gray(img_np)
    resized = utils.resize_to_rectangle(gray, screen_template)
    cv2.imshow("frame", resized)
    key = cv2.waitKey(1)
    if key == 27:
        break
