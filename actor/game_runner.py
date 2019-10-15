import pyautogui

from actor.game_simulation import GameSimulation


class GameRunner:
    def __init__(self, game_simulation: GameSimulation):
        self.game_simulation = game_simulation
        self.terminated = False

    def start(self):
        pyautogui.press('enter')
        self.game_simulation.start()

    def play(self, full_gray_np):
        result = self.game_simulation.play(full_gray_np)
        if result == 'jump':
            pyautogui.press('space')
        elif result == 'down':
            pyautogui.press('down')
        elif result == 'terminate':
            self.terminated = True

    def get_score(self):
        return self.game_simulation.end_time - self.game_simulation.start_time
