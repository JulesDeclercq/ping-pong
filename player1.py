"""
Hoofd editors: Kevin en Bram

AI controller for paddle1 (left side).

This player uses CSV-based learning to map paddle positions to shot parameters
(strength, angle, spin). The learning system updates after successful shots.
"""
from typing import Tuple
import pybullet as pyb
import numpy as np
from learning import LearningPlayer
import environment as env

# Initialize learning system (will use config from env if available)
def _create_learner():
    exploitation = getattr(env, 'EXPLOITATION_RATIO', 0.88)
    diff_1 = getattr(env, 'DIFFUSION_RATIO_1', 0.66)
    diff_2 = getattr(env, 'DIFFUSION_RATIO_2', 0.44)
    return LearningPlayer('paddle1', 'paddle1.csv', exploitation, diff_1, diff_2)

_learner = _create_learner()
_last_position = None

# Default shot parameters (used as baseline)
DEFAULT_PARAMS = {'strength': 1.0, 'angle_deg': 0.0, 'spin_deg': 0.0}


def get_shot_params(x: float, y: float) -> Tuple[float, float, float]:
    """Get shot parameters for current paddle position using learned values.
    
    Args:
        x, y: current paddle position
    
    Returns:
        (strength, angle_deg, spin_deg)
    """
    global _last_position
    _last_position = (round(float(x), 1), round(float(y), 1))
    strength, angle_deg, spin_deg = _learner.get_shot_params(x, y)
    return strength, angle_deg, spin_deg


def record_success(strength: float = None, angle_deg: float = None, spin_deg: float = None):
    """Record a successful shot in the learning system.

    If explicit parameters are provided, use them. Otherwise fall back to the
    last shot produced by the learner (so we store the actual action chosen
    at the time `get_shot_params` was called).
    """
    global _last_position
    if strength is None or angle_deg is None or spin_deg is None:
        last = getattr(_learner, '_last_shot', None)
        if last is None:
            return
        strength = last['strength']
        angle_deg = last['angle_deg']
        spin_deg = last['spin_deg']

    if _last_position:
        print(f"[PLAYER1] recording success for pos {_last_position} -> {strength:.3f},{angle_deg:.2f},{spin_deg:.2f}")
        _learner.record_success(_last_position[0], _last_position[1], strength, angle_deg, spin_deg)


def reset_for_new_rally():
    """Reset player state for a new rally."""
    global _last_position
    _last_position = None
