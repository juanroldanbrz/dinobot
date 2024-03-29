import numpy as np
import pyscreenshot as ImageGrab

from component import enemy_detector, game_status
from component.game_runner import GameRunner
from service import learning_model_service, generation_service
from service.learning_model_service import update_score
from template.template import screen_template, enemy_segment_template
from utils import utils

generation = learning_model_service.get_last_generation()
print('------------')
print(f'Starting generation {generation}')
model = learning_model_service.find_one(processed=False, generation=generation)
game_runner = GameRunner(model)
print(f'Loading model: {model.model_id}')
game_runner.start()

phase = 1
i = 0
while True:
    img = ImageGrab.grab(bbox=screen_template.to_tuple(), childprocess=False)
    img_np = np.array(img)
    img_np = utils.resize_to_rectangle(img_np, screen_template)

    full_gray_np = utils.to_gray(img_np)

    # Detect enemies
    phase = game_status.get_phase(full_gray_np)

    # now = time()
    #
    # if now - start_time > 55:
    #     print('Printing snapshot')
    #     cv2.imwrite(f'save/{i}.png', full_gray_np)
    #     i += 1

    enemies, _ = enemy_detector.find_enemies(utils.to_gray(img_np), phase)

    game_runner.play(enemies, enemy_segment_template.shape())
    game_status_str = game_status.get_game_status(utils.to_gray(img_np))

    if game_status_str == 'game_over':
        game_runner.terminate()

        print(f'elapsed: {game_runner.get_score()}')
        print(f'{learning_model_service.count_non_processed(generation)} '
              f'remaining models from {generation} generation')

        update_score(model.model_id, generation, game_runner.get_score())

        if learning_model_service.count_non_processed(generation) == 0:
            print('-----------')
            print(f'Generation {generation} finished. Creating generation {generation + 1}')
            generation_service.create_generation_report(generation)
            generation_service.reproduce_generation(generation)
            generation = generation + 1

        model = learning_model_service.find_one(processed=False, generation=generation)
        game_runner = GameRunner(model)
        print(f'Loading model: {model.model_id}, generation: {generation}')
        game_runner.start()
