import cv2
import numpy as np
from config import Config
from utils import get_cv2_xy


class Missile():

    ORIGIN_X = 0.0
    ORIGIN_Y = 0.0

    def __init__(self):
        pass

    def launch(self, target):

        if target.x == self.ORIGIN_X and target.y == self.ORIGIN_Y:
            vx, vy = 0.0, 0.0

        else:
            norm = np.sqrt(np.square(target.x) + np.square(target.y))

            # Unit vector
            ux = target.x / norm
            uy = target.y / norm

            # Speed
            vx = Config.MISSILES.SPEED * ux
            vy = Config.MISSILES.SPEED * uy

        new_missile = np.array([[self.ORIGIN_X, self.ORIGIN_Y, target.x,
                target.y, vx, vy]], dtype=np.float32
        )

        self.missile_movement = np.vstack(
            (self.missile_movement, new_missile))

    def reset(self, seed=None):
        self.missile_movement = np.zeros((0, 6), dtype=np.float32)
        self.missile_explosion = np.zeros((0, 3), dtype=np.float32)

    def step(self, action):
        """
             Args:
                action (int) : {0: do nothing, 1: target up, 
                2: target down, 3: target left, 
                4: target right, 5: fire missile}
            
            returns:
                observation: None.
                reward: None.
                done: None.
                info: None.
        
        """
        
        # Compute horizontal and vertical distances to target
        
        dx = np.abs(
            self.missile_movement[:, 2] - self.missile_movement[:, 0])
        dy = np.abs(
            self.missile_movement[:, 3] - self.missile_movement[:, 1])
        
        movement_x = np.minimum(np.abs(self.missile_movement[:, 4]), dx)
        movement_y = np.minimum(np.abs(self.missile_movement[:, 5]), dy)
        
        movement_x *= np.sign(self.missile_movement[:, 4])
        movement_y *= np.sign(self.missile_movement[:, 5])
        
        self.missile_movement[:, 0] +=movement_x
        self.missile_movement[:, 1] +=movement_y
        
        self.missile_explosion[:,2] += Config.MISSILES.EXPLOSION_SPEED
        
        # Indices of new exploding missiles
        new_exploding_missiles_indices = np.argwhere(
            (self.missile_movement[:, 0] == self.missile_movement[:, 2]) &
            (self.missile_movement[:, 1] == self.missile_movement[:, 3])
        )
        nb_new_exploding_missiles = new_exploding_missiles_indices.shape[0]
        # print(new_exploding_missiles_indices)
        
        if nb_new_exploding_missiles > 0:
            new_exploding_missiles_indices = np.squeeze(
                new_exploding_missiles_indices)

            # Get positions
            x = self.missile_movement[new_exploding_missiles_indices, 0]
            y = self.missile_movement[new_exploding_missiles_indices, 1]

            # Remove missiles
            self.missile_movement = np.delete(
                self.missile_movement, new_exploding_missiles_indices, axis=0)

            # Create new ones
            new_exploding_missiles = np.zeros(
                (nb_new_exploding_missiles, 3),
                dtype=np.float32,
            )

            # Affect positions
            new_exploding_missiles[:, 0] = x
            new_exploding_missiles[:, 1] = y

            # Add them
            self.missile_explosion = np.vstack(
                (self.missile_explosion, new_exploding_missiles))

        # Remove missiles with full explosion

        full_explosion_indices = np.squeeze(np.argwhere(
            (self.missile_explosion[:, 2] >
                Config.MISSILES.EXPLOSION_RADIUS)
        ))

        self.missile_explosion = np.delete(
            self.missile_explosion, full_explosion_indices, axis=0)

        return None, None, None, None
    
    def render(self, observation):
        for  x,y in zip(self.missile_movement[:, 0], 
                        self.missile_movement[:, 1]):
            cv2.line(
                img = observation,
                pt1=(get_cv2_xy(Config.EPISODE.HEIGHT,
                                Config.EPISODE.WIDTH,
                                0.0,
                                0.0)),
                pt2=(get_cv2_xy(Config.EPISODE.HEIGHT,
                                Config.EPISODE.WIDTH,
                                x,
                                y)),
                color=Config.COLORS.MISSILE,
                thickness=2,
            )
            
            cv2.circle(
                img = observation,
                center= (get_cv2_xy(Config.EPISODE.HEIGHT,
                                    Config.EPISODE.WIDTH,
                                    x,
                                    y)),
                radius=int(Config.MISSILES.RADIUS),
                color = Config.COLORS.MISSILE,
                thickness= -1,
            )
        for x, y, explosion in zip(self.missile_explosion[:, 0],
                                   self.missile_explosion[:, 1],
                                   self.missile_explosion[:, 2]):
            cv2.circle(
                img=observation,
                center=(get_cv2_xy(Config.EPISODE.HEIGHT,
                                   Config.EPISODE.WIDTH,
                                   x,
                                   y)),
                radius=int(explosion),
                color=Config.COLORS.EXPLOSION,
                thickness=-1,
            )
                    