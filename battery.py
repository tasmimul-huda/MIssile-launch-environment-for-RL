import cv2
import numpy as np
from config import Config
from utils import get_cv2_xy


class Battery():
    """
    NB_BATTERY (int): total # batteries. Only one in this case
    """

    NB_BATTERIES = 1

    def __init__(self):
        self.battery = np.zeros((self.NB_BATTERIES, 1), dtype=np.float32)

    def reset(self, seed=None):
        """
        Reset Battery.i.e total # missiles is reset to default
        """
        self.battery[:, 0] = Config.MISSILES.NUMBER
        self.nb_missiles_launched = 0

    def step(self, action):
        """
        To Go to next state from current state we use the step function

            Args:
                action (int) : {0: do nothing, 1: target up, 
                2: target down, 3: target left, 
                4: target right, 5: fire missile}

            Returns:
                observation: None, reward: None, done: None, 
                info: info (dict): additional information of the current time step. It
                contains key "can_fire" with associated value "True" if the
                anti-missile battery can fire a missile and "False" otherwise.
        """
        can_fire = self.battery[0, 0] > 0
        if action == 5 and can_fire:
            self.battery[0, 0] -= 1

        return None, None, None, {'can_fire': can_fire}

    def render(self, observation):
        cv2.circle(
            img=observation,
            center=(get_cv2_xy(Config.EPISODE.HEIGHT,
                               Config.EPISODE.WIDTH,
                               0.0, 0.0)),
            radius=int(Config.BATTERY.RADIUS),
            color=Config.COLORS.BATTERY,
            thickness=-1
        )
