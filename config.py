import numpy as np
from dataclasses import dataclass


class Config():

    @dataclass
    class EPISODE():
        FPS: int = 60
        HEIGHT: int = 200
        WIDTH: int = 200

    # Missile Battery
    @dataclass
    class BATTERY():
        RADIUS: float = 25.0

    @dataclass
    class COLORS():
        BACKGROUND: tuple = (255, 255, 255)
        BATTERY: tuple = (0, 0, 0)
        ENEMY_MISSILE: tuple = (255, 0, 0)
        EXPLOSION: tuple = (226, 88, 34)
        # FRIENDLY_MISSILES: tuple = (0, 255, 0)
        MISSILE: tuple = (0, 255, 0)
        TARGET: tuple = (0, 0, 255)

    @dataclass
    class ENEMIES():  # ENEMY_MISSILES
        NUMBER: int = 19
        PROBA_IN: float = 0.005
        RADIUS: float = 4.0
        SPEED: float = 1.0

    @dataclass
    class MISSILES():
        NUMBER: int = 1400
        EXPLOSION_RADIUS: float = 20.0
        EXPLOSION_SPEED: float = 0.5
        RADIUS: float = 7.0
        SPEED: float = 7.0

    @dataclass
    class OBSERVATION():
        HEIGHT: float = 84
        RENDER_PROCESSED_HEIGHT: int = 250
        RENDER_PROCESSED_WIDTH: int = 250
        WIDTH: float = 84

    @dataclass
    class REWARD():
        DESTROYED_ENEMEY: float = 20.0
        MISSILE_LAUNCHED: float = -6.0

    @dataclass
    class TARGET():
        SIZE: int = 12
        VX: int = 4
        VY: int = 4
