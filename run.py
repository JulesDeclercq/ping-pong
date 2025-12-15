"""
Hoofd editors: Kevin

Small runner for the ping-pong environment.
Edit the variables below to point to your player modules, set the desired
winning score (set <=0 for infinite), and choose the animation speed.

This script will dynamically load player modules from file paths and
inject them into sys.modules as 'player1' and 'player2' so the environment
can import them normally.
"""
import os
import sys
import importlib.util

# ------------------------ USER CONFIG ------------------------
# Edit these paths (relative to this file or absolute)
PLAYER1_PATH = "player1.py"
PLAYER2_PATH = "player2.py"

# Winning score: first to reach this score ends the match. Set <= 0 to disable.
WINNING_SCORE = -1

# Animation speed: frames per second for ball animation. Higher = faster.
ANIMATION_FPS = 240

# Start PyBullet GUI? Set to False if running headless (may still require X server)
START_GUI = False

# Whether to enable keyboard input polling (not needed if you only use parameterized players)
ENABLE_INPUT = False

# Game mode: "PLAY" for normal gameplay with animation, "LEARNING" for fast learning (no animation)
# Default game mode: PLAY (animated, interactive). Set to "LEARNING" for fast headless training.
GAME_MODE = "LEARNING"

# Learning parameters
EXPLOITATION_RATIO = 0.88  # Percentage (0.0-1.0) to use CSV values vs random exploration
DIFFUSION_RATIO_1 = 0.66   # How similar ±1 range is to center (0.0-1.0)
DIFFUSION_RATIO_2 = 0.44   # How similar ±2 range is to center (0.0-1.0)
# -------------------------------------------------------------

import environment as env


def _load_player_module(path: str, mod_name: str):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Player file not found: {path}")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    # insert into sys.modules so `import player1` works inside environment
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    print("Run: loading players and starting environment")

    p1 = _load_player_module(PLAYER1_PATH, 'player1')
    p2 = _load_player_module(PLAYER2_PATH, 'player2')

    # If players expose a params dict, use it to populate environment.PLAYER_PARAMS
    try:
        p1_params = getattr(p1, 'DEFAULT_PARAMS', None) or getattr(p1, 'PLAYER_PARAMS', None) or getattr(p1, 'PARAMS', None)
        if isinstance(p1_params, dict):
            env.PLAYER_PARAMS['paddle1'].update(p1_params)
    except Exception:
        pass
    try:
        p2_params = getattr(p2, 'DEFAULT_PARAMS', None) or getattr(p2, 'PLAYER_PARAMS', None) or getattr(p2, 'PARAMS', None)
        if isinstance(p2_params, dict):
            env.PLAYER_PARAMS['paddle2'].update(p2_params)
    except Exception:
        pass

    # Apply runner configuration
    env.WINNING_SCORE = WINNING_SCORE
    env.LEARNING_MODE = (GAME_MODE == "LEARNING")
    env.EXPLOITATION_RATIO = EXPLOITATION_RATIO
    env.DIFFUSION_RATIO_1 = DIFFUSION_RATIO_1
    env.DIFFUSION_RATIO_2 = DIFFUSION_RATIO_2

    # Replace CFG with a new instance to modify ANIMATION_FPS (Config is frozen)
    try:
        cfg = env.Config()
        # create a new Config object copying old values but with updated ANIMATION_FPS
        cfg_values = {f.name: getattr(env.CFG, f.name) for f in type(env.CFG).__dataclass_fields__.values()}
        cfg_values['ANIMATION_FPS'] = ANIMATION_FPS
        env.CFG = env.Config(**cfg_values)
    except Exception:
        # if for some reason we can't replace CFG, ignore and proceed
        pass

    # Start environment (do not start input polling unless explicitly enabled)
    env.start(start_input=ENABLE_INPUT)

    # Play rallies indefinitely until interrupted (WINNING_SCORE = -1 means no end condition)
    try:
        rally_count = 0
        while True:
            env.rally()
            rally_count += 1
            # Optionally print progress every 100 rallies
            if rally_count % 100 == 0:
                print(f"[Rallies: {rally_count}] Score: {env.SCORE}")
    except KeyboardInterrupt:
        print(f"\n[Training stopped] Total rallies: {rally_count}")
    finally:
        print(f"Final score: {env.SCORE}")
        env.stop()
