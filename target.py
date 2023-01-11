import cv2
from config import Config
from utils import get_cv2_xy


class Target:
    """
        x (float) : x positon
        y (float) : y position
    """

    def __init__(self):
        pass

    def reset(self, seed=None):
        self.x = 0.0
        self.y = Config.EPISODE.HEIGHT / 2

    def step(self, action):
        """
        To Go to next state from current state we use the step function

            Args:
                action (int) : {0: do nothing, 1: target up, 
                2: target down, 3: target left, 
                4: target right, 5: fire missile}

            Returns:
                observation: None, reward: None, done: None, info: None
        """
        if action == 1:
            self.y = min(Config.EPISODE.HEIGHT, self.y + Config.TARGET.VY)
        elif action == 2:
            self.y = max(0, self.y - Config.TARGET.VY)
        elif action == 3:
            self.x = max(-Config.EPISODE.WIDTH / 2, self.x - Config.TARGET.VX)
        elif action == 4:
            self.x = min(Config.EPISODE.WIDTH / 2, self.x + Config.TARGET.VX)

        return None, None, None, None

    def render(self, observation):
        """
        Render Target
        The target is a cross, represented by 4 coordinates, 2 for the
        horizontal line and 2 for the vertical line.

        """
        # Horizontal
        cv2.line(
            img=observation,
            pt1=(get_cv2_xy(Config.EPISODE.HEIGHT,
                            Config.EPISODE.WIDTH,
                            self.x - Config.TARGET.SIZE,
                            self.y)),
            pt2=(get_cv2_xy(Config.EPISODE.HEIGHT,
                            Config.EPISODE.WIDTH,
                            self.x + Config.TARGET.SIZE,
                            self.y)),
            color=Config.COLORS.TARGET,
            thickness=2,
        )

        # Vertical
        cv2.line(
            img=observation,
            pt1=(get_cv2_xy(Config.EPISODE.HEIGHT,
                            Config.EPISODE.WIDTH,
                            self.x,
                            self.y + Config.TARGET.SIZE)),
            pt2=(get_cv2_xy(Config.EPISODE.HEIGHT,
                            Config.EPISODE.WIDTH,
                            self.x,
                            self.y - Config.TARGET.SIZE)),
            color=Config.COLORS.TARGET,
            thickness=3,
        )

        # cv2.circle(
        #     img=observation,
        #     center=(get_cv2_xy(Config.EPISODE.HEIGHT,
        #                        Config.EPISODE.WIDTH,
        #                        self.x,
        #                        self.y + Config.TARGET.SIZE)),
        #     radius=int(Config.BATTERY.RADIUS),
        #     color=Config.COLORS.TARGET,
        #     thickness=1
        # )
