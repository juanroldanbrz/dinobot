import cv2

from template.template import screen_template, game_over_template, phase_recognition_template
from utils import utils
from utils.assertions import assert_rectangle_shape, assert_gray_img

font = cv2.FONT_HERSHEY_COMPLEX


def _find_rectangle(gray_np, expected_area: ()) -> bool:
    _, threshold = cv2.threshold(gray_np, 240, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray_np, 100, 200)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if expected_area[0] < area < expected_area[1]:
            return True
    return False


def get_phase(full_gray_np) -> int:
    assert_rectangle_shape(full_gray_np, screen_template, f'Shape of the image should be {screen_template.shape()}')
    assert_gray_img(full_gray_np)
    roi_gray_np = utils.crop_image(full_gray_np, phase_recognition_template)

    flat_pixels = roi_gray_np.ravel()
    all_pixels_average = sum(flat_pixels) / len(flat_pixels)
    if all_pixels_average > 150:
        # Phase 1
        return 1
    else:
        return 2


def get_game_status(full_gray_np) -> str:
    # Assert
    assert_rectangle_shape(full_gray_np, screen_template, f'Shape of the image should be {screen_template.shape()}')
    assert_gray_img(full_gray_np)

    game_over_img_gray = utils.crop_image(full_gray_np, game_over_template)
    if _find_rectangle(game_over_img_gray, (1200, 1250)):
        return 'game_over'
    else:
        return 'playing'
