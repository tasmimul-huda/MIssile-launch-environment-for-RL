import cv2
import gym
import numpy as np
import pygame as pg
from gym import spaces

from config import Config
from battery import Battery
from target import Target
from missile import Missile
from enemy import Enemy


class MissileEnv(gym.Env):

    NB_ACTIONS = 6

    def __init__(self):
        self.action_space = spaces.Discrete(self.NB_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(Config.OBSERVATION.HEIGHT,
                                                   Config.OBSERVATION.WIDTH, 3),
                                            dtype=np.uint8,
                                            )
        self.reward = (-float("inf"), float("inf"))

        # Object initialization
        self.battery = Battery()
        self.missile = Missile()
        self.target = Target()
        self.enemy = Enemy()

        # No display while no render
        self._clock = None
        self._display = None

    def _collisions_missiles(self):

        friendly_exploding = self.missile.missile_explosion

        # Enemy missiles current positions
        enemy_missiles = self.enemy.enemy_missiles[:, [2, 3]]

        # Align enemy missiles and friendly exploding ones
        enemy_m_dup = np.repeat(enemy_missiles,
                                friendly_exploding.shape[0],
                                axis=0)
        friendly_e_dup = np.tile(friendly_exploding,
                                 reps=[enemy_missiles.shape[0], 1])

        # Compute distances
        dx = friendly_e_dup[:, 0] - enemy_m_dup[:, 0]
        dy = friendly_e_dup[:, 1] - enemy_m_dup[:, 1]
        distances = np.sqrt(np.square(dx) + np.square(dy))

        # Get enemy missiles inside an explosion radius
        inside_radius = distances <= (
            friendly_e_dup[:, 2] + Config.ENEMIES.RADIUS)
        inside_radius = inside_radius.astype(int)
        inside_radius = np.reshape(
            inside_radius,
            (enemy_missiles.shape[0], friendly_exploding.shape[0]),
        )

        # Remove these missiles
        missiles_out = np.argwhere(np.sum(inside_radius, axis=1) >= 1)
        self.enemy.enemy_missiles = np.delete(
            self.enemy.enemy_missiles,
            np.squeeze(missiles_out),
            axis=0,
        )

        # Compute current reward
        nb_missiles_destroyed = missiles_out.shape[0]
        self.reward += Config.REWARD.DESTROYED_ENEMEY * \
            nb_missiles_destroyed

    def _compute_observation(self):
        self.observation = np.zeros(
            (Config.EPISODE.WIDTH, Config.EPISODE.HEIGHT, 3), dtype=np.uint8)
        self.observation[:, :, 0] = Config.COLORS.BACKGROUND[0]
        self.observation[:, :, 1] = Config.COLORS.BACKGROUND[1]
        self.observation[:, :, 2] = Config.COLORS.BACKGROUND[2]

        # object drawing
        self.battery.render(self.observation)
        self.missile.render(self.observation)
        self.enemy.render(self.observation)
        self.target.render(self.observation)

    def _process_observation(self):
        processed_observation = cv2.resize(
            self.observation,
            (Config.OBSERVATION.HEIGHT, Config.OBSERVATION.WIDTH),
            interpolation=cv2.INTER_AREA,
        )
        return processed_observation.astype(np.float32)

    def reset(self, seed=None):
        # return observation
        self.time_step = 0
        self.reward_total = 0.0
        self.reward = 0.0

        # Object reset
        self.battery.reset(seed=seed)
        self.missile.reset(seed=seed)
        self.enemy.reset(seed=seed)
        self.target.reset(seed=seed)
        self._compute_observation()

        return self._process_observation()

    def step(self, action):        # return observation, reward, done, {}
        self.reward = 0.0

        # Step functions
        _, battery_reward, _, can_fire_dict = self.battery.step(action)
        _, _, _, _ = self.missile.step(action)
        _, _, done_enemies, _ = self.enemy.step(action)
        _, _, _, _ = self.target.step(action)

        # Launch a new missile
        if action == 5 and can_fire_dict["can_fire"]:
            self.missile.launch(self.target)
            self.reward += Config.REWARD.MISSILE_LAUNCHED
        # Check for collisions
        self._collisions_missiles()
        # Check if episode is finished
        done = done_enemies

        # Compute observation
        self._compute_observation()

        # Update values
        self.time_step += 1
        self.reward_total += self.reward

        return self._process_observation(), self.reward, done, {}

    def render(self, mode="raw_observation"):
        # print("Initializing Screen")

        # width and height for screen render
        W, H = Config.EPISODE.WIDTH, Config.EPISODE.HEIGHT

        if self._display is None:
            pg.init()
            pg.mouse.set_visible(False)
            self._clock = pg.time.Clock()
            pg.display.set_caption("Missile Launcher")
            self._display = pg.display.set_mode((W, H))

        if mode == "processed_observation":
            observation = self._process_observation()
            surface = pg.surfarray.make_surface(observation)
            surface = pg.transform.scale(surface, (H, W))
        else:
            observation = self.observation
            surface = pg.surfarray.make_surface(observation)

        # Display Everything

        self._display.blit(surface, (0, 0))
        pg.display.update()

        # Limiting mas FPS
        self._clock.tick(Config.EPISODE.FPS)

        def close(self):
            if self._display is not None:
                pg.quit()
