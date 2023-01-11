"""Enemy missiles."""

from random import Random

import cv2
import numpy as np

from config import Config
from utils import get_cv2_xy


class Enemy():

    def __init__(self):
        """Initialize missiles."""
        pass

    def _launch_missile(self):

        x0 = self._rng_python.uniform(
            -0.5 * Config.EPISODE.WIDTH, 0.5 * Config.EPISODE.WIDTH)
        y0 = Config.EPISODE.HEIGHT

        # Final position
        x1 = self._rng_python.uniform(
            -0.5 * Config.EPISODE.WIDTH, 0.5 * Config.EPISODE.WIDTH)
        y1 = 0.0

        # Compute speed vectors
        # ------------------------------------------

        # Compute norm
        norm = np.sqrt(np.square(x1 - x0) + np.square(y1 - y0))

        # Compute unit vectors
        ux = (x1 - x0) / norm
        uy = (y1 - y0) / norm

        # Compute speed vectors
        vx = Config.ENEMIES.SPEED * ux
        vy = Config.ENEMIES.SPEED * uy

        # Add the new missile
        # ------------------------------------------

        # Create the missile
        new_missile = np.array(
            [[x0, y0, x0, y0, x1, y1, vx, vy]],
            dtype=np.float32,
        )

        # Add it to the others
        self.enemy_missiles = np.vstack(
            (self.enemy_missiles, new_missile))

        # Increase number of launched missiles
        self.nb_missiles_launched += 1

    def reset(self, seed=None):
        self.enemy_missiles = np.zeros((0, 8), dtype=np.float32)
        self.nb_missiles_launched = 0

        # Create random numbers generator
        self._rng_python = Random(seed)

    def step(self, action):
        """Go from current step to next one.

        - 0) Moving missiles.
        - 1) Potentially launch a new missile.
        - 2) Remove missiles that hit the ground.
        returns:
            observation: None.

            reward: None.

            done (bool): True if the episode is finished, i.d. there are no
                more enemy missiles in the environment and no more enemy
                missiles to be launch. False otherwise.

            info: None.
        """
        # Compute horizontal and vertical distances to targets
        dx = np.abs(self.enemy_missiles[:, 4] - self.enemy_missiles[:, 2])
        dy = np.abs(self.enemy_missiles[:, 5] - self.enemy_missiles[:, 3])

        # Take the minimum between the actual speed and the distance to target
        movement_x = np.minimum(np.abs(self.enemy_missiles[:, 6]), dx)
        movement_y = np.minimum(np.abs(self.enemy_missiles[:, 7]), dy)

        # Keep the right sign
        movement_x *= np.sign(self.enemy_missiles[:, 6])
        movement_y *= np.sign(self.enemy_missiles[:, 7])

        # Step t to step t+1
        self.enemy_missiles[:, 2] += movement_x
        self.enemy_missiles[:, 3] += movement_y

        if self.nb_missiles_launched < Config.ENEMIES.NUMBER:
            if self._rng_python.random() <= Config.ENEMIES.PROBA_IN:
                self._launch_missile()

        missiles_out_indices = np.squeeze(np.argwhere(
            (self.enemy_missiles[:, 2] == self.enemy_missiles[:, 4]) &
            (self.enemy_missiles[:, 3] == self.enemy_missiles[:, 5])
        ))

        self.enemy_missiles = np.delete(
            self.enemy_missiles, missiles_out_indices, axis=0)

        done = self.enemy_missiles.shape[0] == 0 and \
            self.nb_missiles_launched == Config.ENEMIES.NUMBER
        return None, None, done, None

    def render(self, observation):

        for x0, y0, x, y in zip(self.enemy_missiles[:, 0],
                                self.enemy_missiles[:, 1],
                                self.enemy_missiles[:, 2],
                                self.enemy_missiles[:, 3]):
            cv2.line(
                img=observation,
                pt1=(get_cv2_xy(Config.EPISODE.HEIGHT,
                                Config.EPISODE.WIDTH,
                                x0,
                                y0)),
                pt2=(get_cv2_xy(Config.EPISODE.HEIGHT,
                                Config.EPISODE.WIDTH,
                                x,
                                y)),
                color=Config.COLORS.ENEMY_MISSILE,
                thickness=1,
            )

            cv2.circle(
                img=observation,
                center=(get_cv2_xy(Config.EPISODE.HEIGHT,
                                   Config.EPISODE.WIDTH,
                                   x,
                                   y)),
                radius=int(Config.ENEMIES.RADIUS),
                color=Config.COLORS.ENEMY_MISSILE,
                thickness=-1,
            )
