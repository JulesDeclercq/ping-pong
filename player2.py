"""
Simple keyboard controller for paddle2 (right side). Mirrored controls.

Usage:
- Call `reset_for_new_rally()` at the start of each rally to reset params.
- Regularly call `update_from_keyboard()` (e.g., in a short loop) to poll
  the PyBullet GUI keyboard and apply changes to paddle position and
  `env_new.PLAYER_PARAMS['paddle2']`.

Controls (when PyBullet GUI has focus):
- Left / Right arrows : move paddle left / right (Y axis)
- Up / Down arrows : move paddle forward / back (X axis)
- Comma / Period : decrease / increase shot strength
- Left Bracket / Right Bracket : decrease / increase shot angle (degrees)
- K / L : decrease / increase spin (degrees)
- Space : reset parameters to defaults
"""
from typing import Tuple
import pybullet as pyb
import numpy as np
import env_new

# defaults
DEFAULT_PARAMS = {'strength': 1.0, 'angle_deg': 0.0, 'spin_deg': 0.0}
MOVE_STEP = 0.02  # meters per key press
STRENGTH_STEP = 0.05
ANGLE_STEP = 1.0  # degrees
SPIN_STEP = 1.0

# Allowed positional ranges for right player (paddle2)
X_MIN = 0.05
X_MAX = env_new.CFG.TABLE_L/2 - 0.05
Y_MIN = -env_new.CFG.TABLE_W/2 + 0.02
Y_MAX = env_new.CFG.TABLE_W/2 - 0.02


def clamp_pos(x: float, y: float) -> Tuple[float, float]:
    x = float(np.clip(x, X_MIN, X_MAX))
    y = float(np.clip(y, Y_MIN, Y_MAX))
    return x, y


def reset_for_new_rally():
    """Reset player params and move paddle to default start pose."""
    env_new.PLAYER_PARAMS['paddle2'] = DEFAULT_PARAMS.copy()
    if env_new._PADDLE2 is not None:
        p = [env_new.CFG.TABLE_L/2 - 0.2, 0.0, env_new.CFG.TABLE_H + 0.1]
        ori = pyb.getQuaternionFromEuler([np.radians(90), np.radians(90), env_new._PADDLE2.base_yaw])
        try:
            pyb.resetBasePositionAndOrientation(env_new._PADDLE2.body_id, p, ori)
        except Exception:
            pass


def update_from_keyboard():
    """Poll PyBullet keyboard events and apply to paddle/params.

    Call this often from a short loop while the GUI is open.
    Returns True if an update was applied, False otherwise.
    """
    events = pyb.getKeyboardEvents()
    if not events:
        return False

    updated = False
    # get current paddle pose
    if env_new._PADDLE2 is None:
        return False
    pos, ori = pyb.getBasePositionAndOrientation(env_new._PADDLE2.body_id)
    x, y, z = pos

    for k, v in events.items():
        if v & pyb.KEY_IS_DOWN == 0:
            continue
        # arrow keys and special keys
        if k == pyb.B3G_LEFT_ARROW:
            y -= MOVE_STEP
            updated = True
        elif k == pyb.B3G_RIGHT_ARROW:
            y += MOVE_STEP
            updated = True
        elif k == pyb.B3G_UP_ARROW:
            x -= MOVE_STEP
            updated = True
        elif k == pyb.B3G_DOWN_ARROW:
            x += MOVE_STEP
            updated = True
        elif k == ord(','):
            env_new.PLAYER_PARAMS['paddle2']['strength'] = float(np.clip(env_new.PLAYER_PARAMS['paddle2'].get('strength', 1.0) - STRENGTH_STEP, 0.0, 3.0))
            updated = True
        elif k == ord('.'):
            env_new.PLAYER_PARAMS['paddle2']['strength'] = float(np.clip(env_new.PLAYER_PARAMS['paddle2'].get('strength', 1.0) + STRENGTH_STEP, 0.0, 3.0))
            updated = True
        elif k == ord('['):
            env_new.PLAYER_PARAMS['paddle2']['angle_deg'] = float(np.clip(env_new.PLAYER_PARAMS['paddle2'].get('angle_deg', 0.0) - ANGLE_STEP, -90.0, 90.0))
            updated = True
        elif k == ord(']'):
            env_new.PLAYER_PARAMS['paddle2']['angle_deg'] = float(np.clip(env_new.PLAYER_PARAMS['paddle2'].get('angle_deg', 0.0) + ANGLE_STEP, -90.0, 90.0))
            updated = True
        elif k == ord('k'):
            env_new.PLAYER_PARAMS['paddle2']['spin_deg'] = float(np.clip(env_new.PLAYER_PARAMS['paddle2'].get('spin_deg', 0.0) - SPIN_STEP, -90.0, 90.0))
            updated = True
        elif k == ord('l'):
            env_new.PLAYER_PARAMS['paddle2']['spin_deg'] = float(np.clip(env_new.PLAYER_PARAMS['paddle2'].get('spin_deg', 0.0) + SPIN_STEP, -90.0, 90.0))
            updated = True
        elif k == pyb.B3G_SPACE:
            reset_for_new_rally()
            updated = True

    if updated:
        x, y = clamp_pos(x, y)
        try:
            pyb.resetBasePositionAndOrientation(env_new._PADDLE2.body_id, [x, y, z], ori)
        except Exception:
            pass
    return updated
