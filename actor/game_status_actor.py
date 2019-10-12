import cv2

from model.actor import Actor
from template.template import screen_template, game_over_template
from utils import utils
from utils.assertions import assert_rectangle_shape, assert_gray_img


def process_game_status(img_np) -> str:
    # Assert
    assert_rectangle_shape(img_np, screen_template, f'Shape of the image should be {screen_template.shape()}')
    assert_gray_img(img_np)

    game_over_img = utils.crop_image(img_np, game_over_template)
    cv2.imwrite('game_over_rectangle.png', game_over_img)
    return 'game_over'


class GameStatusActor(Actor):

    def tell(self, message):
        if message.message_type == 'process_status':
            process_game_status(message.content)
