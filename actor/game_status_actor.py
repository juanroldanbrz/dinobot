import cv2

from model.actor import Actor
from template.template import screen_template, game_over_template
from utils import utils
from utils.assertions import assert_rectangle_shape, assert_gray_img

font = cv2.FONT_HERSHEY_COMPLEX


def find_rectangle(gray_np, expected_area: ()) -> bool:
    _, threshold = cv2.threshold(gray_np, 240, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray_np, 100, 200)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if expected_area[0] < area < expected_area[1]:
            return True
    return False


def process_game_status(gray_np) -> str:
    # Assert
    assert_rectangle_shape(gray_np, screen_template, f'Shape of the image should be {screen_template.shape()}')
    assert_gray_img(gray_np)

    game_over_img_gray = utils.crop_image(gray_np, game_over_template)
    if find_rectangle(game_over_img_gray, (1200, 1250)):
        return 'game_over'
    else:
        return 'playing'


class GameStatusActor(Actor):
    def __init__(self):
        self.game_status = 'unknown'

    def tell(self, message):
        if message.message_type == 'process_status':
            self.game_status = process_game_status(message.content)

    def ask(self, message_type: str):
        if message_type == 'game_status':
            return self.game_status
