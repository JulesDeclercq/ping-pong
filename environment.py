"""
Hoofd editors: Jules en Bram

Lightweight PyBullet ping-pong simulator with deterministic two-period parabolic trajectories.

Run the simulation:
    python -m environment

Or from Python:
    import environment as env
    env.start()
    env.rally()
"""

import logging
import time
import random
import threading
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pybullet as pyb
import pybullet_data

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class Config:
    """Physical parameters for the simulation (ITTF standard table tennis)."""
    # Table dimensions
    TABLE_L: float = 2.74         # length (player-to-player), m
    TABLE_W: float = 1.525        # width (left-right), m
    TABLE_H: float = 0.76         # table surface height, m
    TABLE_THICKNESS: float = 0.05 # m
    
    # Net
    NET_HEIGHT: float = 0.1525    # m
    NET_THICKNESS: float = 0.02   # m
    
    # Ball
    BALL_RADIUS: float = 0.02     # m
    
    # Paddle
    BLADE_LENGTH: float = 0.13    # m
    PADDLE_WIDTH: float = 0.15    # m
    PADDLE_HEIGHT: float = 0.005  # m
    HANDLE_LENGTH: float = 0.06   # m
    
    # Simulation
    PHYSICS_FPS: int = 120        # simulation steps per second
    ANIMATION_FPS: int = 240      # ball animation frame rate
    
    # Noise (set to 0 for no noise)
    NOISE_STRENGTH_SENS: float = 10.0   # noise effect on strength
    NOISE_ANGLE_SENS: float = 100.0      # noise effect on angle (degrees)


CFG = Config()
logger = logging.getLogger("environment")

# ============================================================================
# GLOBAL STATE
# ============================================================================

_CLIENT = None
_SC = None
_TABLE = None
_PADDLE1 = None
_PADDLE2 = None
_BALL = None
_NEXT_SERVER = 1

_INPUT_THREAD = None
_INPUT_THREAD_STOP = None

PLAYER_PARAMS = {
    'paddle1': {'strength': 1.0, 'angle_deg': 0.0, 'spin_deg': 0.0},
    'paddle2': {'strength': 1.0, 'angle_deg': 0.0, 'spin_deg': 0.0},
}

SCORE = {'paddle1': 0, 'paddle2': 0}
WINNING_SCORE = -1  # -1 = infinite, otherwise match ends when a player reaches this score

# Learning mode settings
LEARNING_MODE = False  # If True, skip animation for faster learning
EXPLOITATION_RATIO = 0.88
DIFFUSION_RATIO_1 = 0.66
DIFFUSION_RATIO_2 = 0.44

debug_ids = []
active_collisions = {"table": False, "net": False, "paddle1": False, "paddle2": False}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _distance_xy(a: Tuple, b: Tuple) -> float:
    """Euclidean distance in XY plane."""
    return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))


def clear_debug():
    """Remove all debug visualization lines."""
    global debug_ids
    for did in debug_ids:
        try:
            pyb.removeUserDebugItem(did)
        except Exception:
            pass
    debug_ids = []


# ============================================================================
# PHYSICS OBJECTS (Shape Cache, Table, Paddle, Ball)
# ============================================================================

class ShapeCache:
    """Caches collision and visual shapes to avoid duplication."""
    def __init__(self):
        self._col = {}
        self._vis = {}

    def box_collision(self, half_extents):
        key = ("box", tuple(half_extents))
        if key not in self._col:
            self._col[key] = pyb.createCollisionShape(pyb.GEOM_BOX, halfExtents=half_extents)
        return self._col[key]

    def box_visual(self, half_extents, rgba):
        key = ("vbox", tuple(half_extents), tuple(rgba))
        if key not in self._vis:
            self._vis[key] = pyb.createVisualShape(pyb.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
        return self._vis[key]

    def sphere_collision(self, radius):
        key = ("sphere", float(radius))
        if key not in self._col:
            self._col[key] = pyb.createCollisionShape(pyb.GEOM_SPHERE, radius=radius)
        return self._col[key]

    def sphere_visual(self, radius, rgba):
        key = ("vsphere", float(radius), tuple(rgba))
        if key not in self._vis:
            self._vis[key] = pyb.createVisualShape(pyb.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
        return self._vis[key]


class Table:
    """Table with net and decorative markings."""
    def __init__(self, shape_cache: ShapeCache):
        self.sc = shape_cache
        self.table_id = None
        self.net_id = None

    def create(self):
        c = CFG
        half_extents = [c.TABLE_L / 2, c.TABLE_W / 2, c.TABLE_THICKNESS / 2]
        col = self.sc.box_collision(half_extents)
        vis = self.sc.box_visual(half_extents, [1, 1, 1, 1])
        table_z = c.TABLE_H - c.TABLE_THICKNESS / 2
        self.table_id = pyb.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, 
                                            baseVisualShapeIndex=vis, basePosition=[0, 0, table_z])

        # Decorative blue squares
        square_half = [0.65, 0.3475, 0.001]
        square_vis = self.sc.box_visual(square_half, [0.05, 0.15, 0.45, 1.0])
        offs_x = c.TABLE_L / 2 - square_half[0] - 0.05
        offs_y = c.TABLE_W / 2 - square_half[1] - 0.05
        for sx in (-offs_x, offs_x):
            for sy in (-offs_y, offs_y):
                pyb.createMultiBody(baseMass=0, baseVisualShapeIndex=square_vis,
                                   basePosition=[sx, sy, c.TABLE_H + square_half[2]])

        # Legs
        leg_half = [0.02, 0.02, 0.4]
        leg_col = self.sc.box_collision(leg_half)
        leg_vis = self.sc.box_visual(leg_half, [0.1, 0.1, 0.1, 1.0])
        leg_x = c.TABLE_L / 2 - 0.1
        leg_y = c.TABLE_W / 2 - 0.1
        leg_z = c.TABLE_H - c.TABLE_THICKNESS / 2 - leg_half[2]
        for sx in (-leg_x, leg_x):
            for sy in (-leg_y, leg_y):
                pyb.createMultiBody(baseMass=0, baseCollisionShapeIndex=leg_col,
                                   baseVisualShapeIndex=leg_vis, basePosition=[sx, sy, leg_z])

        # Net
        net_half = [c.NET_THICKNESS / 2, c.TABLE_W / 2, c.NET_HEIGHT / 2]
        net_col = self.sc.box_collision(net_half)
        net_vis = self.sc.box_visual(net_half, [0.0, 0.0, 0.0, 1.0])
        self.net_id = pyb.createMultiBody(baseMass=0, baseCollisionShapeIndex=net_col,
                                         baseVisualShapeIndex=net_vis,
                                         basePosition=[0, 0, c.TABLE_H + net_half[2]])
        return self.table_id


class Paddle:
    """Paddle with blade and handle."""
    def __init__(self, shape_cache: ShapeCache, base_pos=(0, 0, 0), color=(1, 0, 0, 1)):
        self.sc = shape_cache
        self.base_pos = list(base_pos)
        self.color = list(color)
        self.body_id = None

    def create(self):
        c = CFG
        blade_half = [c.BLADE_LENGTH / 2, c.PADDLE_WIDTH / 2, c.PADDLE_HEIGHT / 2]
        blade_col = self.sc.box_collision(blade_half)
        blade_vis = self.sc.box_visual(blade_half, self.color)

        handle_half = [c.HANDLE_LENGTH / 2, 0.015 / 2, c.PADDLE_HEIGHT / 2]
        handle_vis = self.sc.box_visual(handle_half, [0.1, 0.1, 0.1, 1.0])

        base_ori = pyb.getQuaternionFromEuler([np.radians(90), np.radians(90), 1.5708])
        self.body_id = pyb.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=blade_col, baseVisualShapeIndex=blade_vis,
            basePosition=self.base_pos, baseOrientation=base_ori,
            linkMasses=[0.0], linkCollisionShapeIndices=[-1], linkVisualShapeIndices=[handle_vis],
            linkPositions=[[0, c.PADDLE_WIDTH / 2 + handle_half[0], 0]],
            linkOrientations=[pyb.getQuaternionFromEuler([0, 0, np.radians(90)])],
            linkInertialFramePositions=[[0, 0, 0]], linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0], linkJointTypes=[pyb.JOINT_FIXED], linkJointAxis=[[0, 0, 0]]
        )
        return self.body_id


class Ball:
    """Simple ball object."""
    def __init__(self, shape_cache: ShapeCache, start_pos=(0, 0, 1.0)):
        self.sc = shape_cache
        self.start_pos = list(start_pos)
        self.body_id = None

    def create(self):
        col = self.sc.sphere_collision(CFG.BALL_RADIUS)
        vis = self.sc.sphere_visual(CFG.BALL_RADIUS, [1, 1, 1, 1])
        self.body_id = pyb.createMultiBody(baseMass=0.0027, baseCollisionShapeIndex=col,
                                          baseVisualShapeIndex=vis, basePosition=self.start_pos)
        return self.body_id

# ============================================================================
# TRAJECTORY GENERATION
# ============================================================================

def get_optimal_hit_point(paddle_obj) -> Tuple:
    """Return the optimal hit point on a paddle."""
    pos, ori = pyb.getBasePositionAndOrientation(paddle_obj.body_id)
    pos = np.array(pos)
    rot = np.array(pyb.getMatrixFromQuaternion(ori)).reshape(3, 3)
    forward = rot.dot(np.array([1.0, 0.0, 0.0]))
    optimal = pos + forward * CFG.BALL_RADIUS
    return tuple(optimal.tolist())


def generate_parabola_points(origin, contact_point, power=0.5, angle_rad=0.0, 
                             anchor_point=None, samples=128, dir_sign=1) -> Tuple[List, Tuple, Tuple]:
    """Generate parabola points: single arc from origin to landing point.
    
    Args:
        origin: start position (x,y,z) at table height
        contact_point: ball hit position (x,y,z)
        power: horizontal travel distance
        angle_rad: rotation angle around anchor
        anchor_point: rotation center (x,y)
        samples: number of points
        dir_sign: +1 or -1 for direction
    
    Returns:
        (points, left_contact, right_contact)
    """
    if anchor_point is None:
        anchor_point = (contact_point[0], contact_point[1])
    ax, ay = anchor_point[0], anchor_point[1]

    table_z = CFG.TABLE_H + CFG.BALL_RADIUS
    D = float(power) if power > 1e-6 else 0.1
    h = D / 2.0
    k = CFG.TABLE_H + (4.0 / 3.0) * CFG.NET_HEIGHT

    z0 = float(origin[2])
    a = (z0 - k) / ((0 - h) ** 2) if (0 - h) ** 2 != 0 else -1e-6

    points = []
    for i in range(samples):
        s = (i / float(samples - 1)) * D
        lx = dir_sign * s
        z = a * (s - h) ** 2 + k
        wx = ax + lx * np.cos(angle_rad)
        wy = ay + lx * np.sin(angle_rad)
        points.append((wx, wy, float(z)))

    # Align parabola so a point at contact_point.z matches contact_point.xy
    try:
        contact_z = float(contact_point[2])
        best_idx = min(range(len(points)), key=lambda i: abs(points[i][2] - contact_z))
        px, py, _ = points[best_idx]
        dx_off, dy_off = float(contact_point[0]) - px, float(contact_point[1]) - py
        if abs(dx_off) > 1e-9 or abs(dy_off) > 1e-9:
            points = [(x + dx_off, y + dy_off, z) for x, y, z in points]
    except Exception:
        pass

    landing = points[-1]
    left_contact = (landing[0] - 0.01, landing[1], table_z)
    right_contact = (landing[0] + 0.01, landing[1], table_z)
    return points, left_contact, right_contact


def generate_shot_parabolas(ball_pos, strength=1.0, angle_deg=0.0, spin_deg=0.0, 
                           samples=128) -> Tuple[List, List, dict]:
    """Generate two-period trajectory: serve + bounce."""
    bx, by, _ = ball_pos
    angle_rad = float(np.radians(angle_deg))
    spin_rad = float(np.radians(spin_deg))
    table_z = CFG.TABLE_H + CFG.BALL_RADIUS

    max_d = 0.75 * CFG.TABLE_L
    d = max_d * float(strength)
    dir_sign = 1 if bx <= 0 else -1

    origin = (bx - 0.02 * dir_sign, by, table_z)
    pts1, _, right1 = generate_parabola_points(origin, contact_point=ball_pos, power=d,
                                               angle_rad=angle_rad, anchor_point=ball_pos,
                                               samples=samples, dir_sign=dir_sign)
    
    bounce = right1
    bounce_anchor = (bounce[0], bounce[1], table_z)
    pts2, _, right2 = generate_parabola_points(bounce_anchor, contact_point=bounce_anchor, power=d,
                                               angle_rad=angle_rad + spin_rad, anchor_point=bounce_anchor,
                                               samples=samples, dir_sign=dir_sign)
    
    return pts1, pts2, {'bounce1': right1, 'bounce2': right2}

# ============================================================================
# COLLISION DETECTION
# ============================================================================

def check_collision_with_table(ball_pos) -> bool:
    """Check if ball is at table surface."""
    table_z = CFG.TABLE_H + CFG.BALL_RADIUS
    if abs(ball_pos[2] - table_z) < 0.001:
        print(f"[TABLE] collision at {tuple(ball_pos)}")
        return True
    return False


def check_collision_with_net(ball_pos) -> bool:
    """Check if ball hits the net."""
    net_x = CFG.NET_THICKNESS / 2
    net_y = CFG.TABLE_W / 2
    net_z = CFG.TABLE_H + CFG.NET_HEIGHT
    
    bx, by, bz = ball_pos
    hit = (abs(bx) < net_x + CFG.BALL_RADIUS and
           abs(by) < net_y + CFG.BALL_RADIUS and
           bz < net_z + CFG.BALL_RADIUS)
    if hit:
        print(f"[NET] collision at {tuple(ball_pos)}")
    return hit


def check_collision_with_paddle(ball_pos, paddle_pos) -> bool:
    """Check if ball collides with paddle blade."""
    half_x = CFG.PADDLE_HEIGHT / 2
    half_y = CFG.PADDLE_WIDTH / 2
    half_z = CFG.BLADE_LENGTH / 2
    
    dx = abs(ball_pos[0] - paddle_pos[0])
    dy = abs(ball_pos[1] - paddle_pos[1])
    dz = abs(ball_pos[2] - paddle_pos[2])
    
    hit = (dx < half_x + CFG.BALL_RADIUS and
           dy < half_y + CFG.BALL_RADIUS and
           dz < half_z + CFG.BALL_RADIUS)
    if hit:
        print(f"[PADDLE] collision at {tuple(ball_pos)}")
    return hit


def _is_point_on_table_surface(ball_pos) -> bool:
    """Return True if a point is on the physical table surface (z ~ table_z) and within XY bounds."""
    table_z = CFG.TABLE_H + CFG.BALL_RADIUS
    bx, by, bz = ball_pos
    on_z = abs(bz - table_z) < 0.001
    on_xy = (abs(bx) <= CFG.TABLE_L / 2 + 1e-6) and (abs(by) <= CFG.TABLE_W / 2 + 1e-6)
    if on_z and on_xy:
        print(f"[TABLE] valid contact at {tuple(ball_pos)}")
        return True
    return False


def _teleport_opponent_to_second_parabola(pts2: List[Tuple], last_hitter: str):
    """Teleport opponent paddle to position on pts2 closest to net, keeping Z fixed.

    Finds point on pts2 closest to net (min abs(x)) and uses its rounded (X, Y).
    Paddle Z height always stays at its current value (no change to height).
    """
    if not pts2:
        return
    opponent = 'paddle2' if last_hitter == 'paddle1' else 'paddle1'
    paddle_obj = _PADDLE2 if opponent == 'paddle2' else _PADDLE1
    if paddle_obj is None:
        return
    try:
        # get current paddle position (especially Z)
        pos, ori = pyb.getBasePositionAndOrientation(paddle_obj.body_id)
        paddle_z = float(pos[2])
        
        # find point on pts2 closest to net (min |x|)
        best_by_x = min(pts2, key=lambda p: abs(p[0]))
        best_x, best_y, best_z = best_by_x
        
        # round X and Y from this point (discretized to 10 cm grid)
        rounded_x = float(round(best_x, 1))
        rounded_y = float(round(best_y, 1))

        # Limit teleport extents:
        # - longitudinal (X): from net (0) back to full table length (baseline)
        #   left player (paddle1) lives on negative X, right player (paddle2) on positive X
        # - lateral (Y): allow paddle lateral movement from left to right to
        #   slightly beyond the physical table width (half-width + margin).
        #   This returns lateral bounds to about the table width on both sides.
        max_long = 1.0 * CFG.TABLE_L
        max_lat = (CFG.TABLE_W / 2.0) + 0.05  # table half-width + 5 cm margin

        # Keep paddles a short distance away from the net: do not allow x to be
        # closer than `min_from_net` meters. This prevents teleporting the paddle
        # flush against x=0 (the net).
        min_from_net = 0.8  # meters from net
        if opponent == 'paddle2':
            # clamp to [min_from_net, max_long]
            clamped_x = min(max(rounded_x, min_from_net), max_long)
        else:
            # paddle1: clamp to [-max_long, -min_from_net]
            clamped_x = max(min(rounded_x, -min_from_net), -max_long)

        clamped_y = min(max(rounded_y, -max_lat), max_lat)

        new_pos = (clamped_x, clamped_y, paddle_z)
        pyb.resetBasePositionAndOrientation(paddle_obj.body_id, new_pos, ori)
        # run a few physics steps to stabilize the teleported body (reduces phasing)
        try:
            for _ in range(4):
                pyb.stepSimulation()
        except Exception:
            pass

        print(f"[TELEPORT] {opponent} -> {new_pos} (requested: {rounded_x, rounded_y})")
        # Prompt the player's policy to compute its shot for this teleported
        # position so that the learner records a _last_shot/_last_position
        # which can later be saved if the shot succeeds.
        try:
            if opponent == 'paddle2':
                import player2
                # call get_shot_params to set internal _last_position/_last_shot
                player2.get_shot_params(clamped_x, clamped_y)
            else:
                import player1
                player1.get_shot_params(clamped_x, clamped_y)
        except Exception:
            pass
    except Exception:
        pass

# ============================================================================
# VISUALIZATION & ANIMATION
# ============================================================================

def draw_parabola(points: List, color=(1, 1, 0), line_width=2):
    """Draw trajectory as debug lines."""
    global debug_ids
    for i in range(len(points) - 1):
        try:
            did = pyb.addUserDebugLine(points[i], points[i + 1], lineColorRGB=color,
                                      lineWidth=line_width, lifeTime=0)
            debug_ids.append(did)
        except Exception:
            pass

# ============================================================================
# ENVIRONMENT CONTROL
# ============================================================================

def create_ball_at(position) -> Ball:
    """Create or reset ball at given position."""
    global _BALL
    if _BALL is not None:
        try:
            pyb.removeBody(_BALL.body_id)
        except Exception:
            pass
    _BALL = Ball(_SC, start_pos=list(position))
    _BALL.create()
    return _BALL


def start(start_input: bool = False):
    """Start the simulation environment.

    Args:
        start_input: if True, start the keyboard input polling thread (disabled by default).
    """
    global _CLIENT, _SC, _TABLE, _PADDLE1, _PADDLE2
    
    cid = pyb.connect(pyb.GUI)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(1.0 / CFG.PHYSICS_FPS)
    
    # Disable HUD
    for flag in [pyb.COV_ENABLE_GUI, pyb.COV_ENABLE_SHADOWS]:
        try:
            pyb.configureDebugVisualizer(flag, 0)
        except Exception:
            pass
    
    sc = ShapeCache()
    tbl = Table(sc)
    tbl.create()
    
    paddle_h = CFG.TABLE_H + 0.1
    p1 = Paddle(sc, base_pos=[-CFG.TABLE_L/2 + 0.2, 0.0, paddle_h], color=(1, 0.2, 0.2, 1))
    p1.create()
    p2 = Paddle(sc, base_pos=[CFG.TABLE_L/2 - 0.2, 0.0, paddle_h], color=(0.2, 0.2, 1.0, 1))
    p2.create()
    
    _CLIENT = cid
    _SC = sc
    _TABLE = tbl
    _PADDLE1 = p1
    _PADDLE2 = p2
    
    # Optionally start background input polling (disabled by default)
    if start_input:
        try:
            _start_input_thread()
        except Exception:
            pass
    
    print("✓ Environment started. Call: environment.rally()")
    return True


def stop():
    """Stop the simulation."""
    global _CLIENT
    try:
        _stop_input_thread()
    except Exception:
        pass
    try:
        pyb.disconnect()
    except Exception:
        pass
    _CLIENT = None
    print("✓ Environment stopped.")


def rally() -> bool:
    """Run a single rally.
    
    Returns:
        True if rally completed normally, False if match ended (winning_score reached)
    """
    global _NEXT_SERVER, _BALL, SCORE, WINNING_SCORE
    
    if _PADDLE1 is None:
        raise RuntimeError("Environment not started. Call start() first.")
    
    # ========== SETUP ==========
    server = 'paddle1' if _NEXT_SERVER == 1 else 'paddle2'
    server_obj = _PADDLE1 if server == 'paddle1' else _PADDLE2
    last_hitter = server
    
    # Reset player controllers
    try:
        import player1
        player1.reset_for_new_rally()
    except Exception:
        pass
    try:
        import player2
        player2.reset_for_new_rally()
    except Exception:
        pass
    
    # Reset paddles to starting positions BEFORE creating the serve so serves originate
    # from the paddles' start poses (fixes follow-up rallies using previous positions)
    try:
        p1_start = [-CFG.TABLE_L/2 + 0.2, 0.0, CFG.TABLE_H + 0.1]
        p1_ori = pyb.getQuaternionFromEuler([np.radians(90), np.radians(90), 1.5708])
        pyb.resetBasePositionAndOrientation(_PADDLE1.body_id, p1_start, p1_ori)
        
        p2_start = [CFG.TABLE_L/2 - 0.2, 0.0, CFG.TABLE_H + 0.1]
        p2_ori = pyb.getQuaternionFromEuler([np.radians(90), np.radians(90), 1.5708])
        pyb.resetBasePositionAndOrientation(_PADDLE2.body_id, p2_start, p2_ori)
    except Exception:
        pass
    
    # Create ball at server's optimal position (now that paddles are at start)
    start_pt = get_optimal_hit_point(server_obj)
    create_ball_at(start_pt)
    
    print(f"\n>>> Rally started (server={server})")
    
    # ========== RALLY LOOP ==========
    rally_ended = False
    temp_params = None
    hit_pos = None
    
    while not rally_ended:
        # Current ball position
        ball_pos = tuple(hit_pos) if hit_pos is not None else pyb.getBasePositionAndOrientation(_BALL.body_id)[0]
        
        # Get shot parameters (from learning system or base params)
        if temp_params is not None:
            params = temp_params
            temp_params = None
            hit_pos = None
        else:
            params = PLAYER_PARAMS[last_hitter]
        
        base_strength = float(np.clip(params.get('strength', 1.0), 0.4, 1.6))
        base_angle = float(np.clip(params.get('angle_deg', 0.0), -33.0, 33.0))
        base_spin = float(np.clip(params.get('spin_deg', 0.0), -12.0, 12.0))
        
        # Generate trajectories
        pts1, pts2, meta = generate_shot_parabolas(ball_pos, strength=base_strength, 
                                                   angle_deg=base_angle, spin_deg=base_spin)
        
        # Get current paddle positions
        p1_pos = np.array(pyb.getBasePositionAndOrientation(_PADDLE1.body_id)[0])
        p2_pos = np.array(pyb.getBasePositionAndOrientation(_PADDLE2.body_id)[0])
        
        # Find nearest start point in first parabola
        start_idx = min(range(len(pts1)), 
                       key=lambda i: np.linalg.norm(np.array(pts1[i][:2]) - np.array(ball_pos[:2])))
        
        # ========== FIRST PARABOLA ==========
        paddle_hit = False
        teleported = False
        last_z = None
        
        for p in pts1[start_idx:]:
            try:
                pyb.resetBasePositionAndOrientation(_BALL.body_id, p, 
                                                    pyb.getQuaternionFromEuler([0, 0, 0]))
                pyb.stepSimulation()
            except Exception:
                rally_ended = True
                break
            
            # Detect peak passage and teleport opponent
            if not teleported and last_z is not None and p[2] < last_z:
                try:
                    _teleport_opponent_to_second_parabola(pts2, last_hitter)
                    p1_pos = np.array(pyb.getBasePositionAndOrientation(_PADDLE1.body_id)[0])
                    p2_pos = np.array(pyb.getBasePositionAndOrientation(_PADDLE2.body_id)[0])
                    teleported = True
                except Exception:
                    pass
            last_z = p[2]
            
            # Net collision
            if check_collision_with_net(p):
                opponent = 'paddle2' if last_hitter == 'paddle1' else 'paddle1'
                SCORE[opponent] += 1
                print(f"[SCORE] {opponent} +1 (net) → {SCORE}")
                rally_ended = True
                break
            
            # Paddle collision
            if check_collision_with_paddle(p, p1_pos) and last_hitter != 'paddle1':
                last_hitter = 'paddle1'
                paddle_hit = True
                hit_pos = p
                break
            if check_collision_with_paddle(p, p2_pos) and last_hitter != 'paddle2':
                last_hitter = 'paddle2'
                paddle_hit = True
                hit_pos = p
                break
            
            time.sleep(0 if LEARNING_MODE else 1.0 / CFG.ANIMATION_FPS)
        
        # Handle paddle hit from first parabola
        if rally_ended or paddle_hit:
            if paddle_hit:
                try:
                    paddle_pos = pyb.getBasePositionAndOrientation(
                        _PADDLE1.body_id if last_hitter == 'paddle1' else _PADDLE2.body_id
                    )[0]
                    if last_hitter == 'paddle1':
                        import player1
                        strength, angle_deg, spin_deg = player1.get_shot_params(paddle_pos[0], paddle_pos[1])
                    else:
                        import player2
                        strength, angle_deg, spin_deg = player2.get_shot_params(paddle_pos[0], paddle_pos[1])
                except Exception:
                    strength, angle_deg, spin_deg = base_strength, base_angle, base_spin
                
                temp_params = {'strength': strength, 'angle_deg': angle_deg, 'spin_deg': spin_deg}
                try:
                    pyb.resetBasePositionAndOrientation(_BALL.body_id, hit_pos, pyb.getQuaternionFromEuler([0, 0, 0]))
                except Exception:
                    pass
                continue
            break
        
        # ========== PRE-CHECK SECOND PARABOLA ==========
        # Ensure the second parabola contains at least one valid table contact within XY bounds.
        will_hit_table = any(_is_point_on_table_surface(p) for p in pts2)
        if not will_hit_table:
            opponent = 'paddle2' if last_hitter == 'paddle1' else 'paddle1'
            SCORE[opponent] += 1
            print(f"[SCORE] {opponent} +1 (second no valid table contact) → {SCORE}")
            rally_ended = True
            break
        
        # ========== SECOND PARABOLA ==========
        paddle_hit = False
        
        for p in pts2:
            try:
                pyb.resetBasePositionAndOrientation(_BALL.body_id, p,
                                                    pyb.getQuaternionFromEuler([0, 0, 0]))
                pyb.stepSimulation()
            except Exception:
                rally_ended = True
                break
            
            # Net collision
            if check_collision_with_net(p):
                opponent = 'paddle2' if last_hitter == 'paddle1' else 'paddle1'
                SCORE[opponent] += 1
                print(f"[SCORE] {opponent} +1 (net) → {SCORE}")
                rally_ended = True
                break
            
            # Paddle collision
            if check_collision_with_paddle(p, p1_pos) and last_hitter != 'paddle1':
                last_hitter = 'paddle1'
                paddle_hit = True
                hit_pos = p
                break
            if check_collision_with_paddle(p, p2_pos) and last_hitter != 'paddle2':
                last_hitter = 'paddle2'
                paddle_hit = True
                hit_pos = p
                break
            
            if _is_point_on_table_surface(p):
                try:
                    _teleport_opponent_to_second_parabola(pts2, last_hitter)
                except Exception:
                    pass
            elif abs(p[2] - (CFG.TABLE_H + CFG.BALL_RADIUS)) < 0.001:
                # Ball at table height but outside table XY bounds -> missed the table
                opponent = 'paddle2' if last_hitter == 'paddle1' else 'paddle1'
                SCORE[opponent] += 1
                print(f"[SCORE] {opponent} +1 (second landed out of table bounds) → {SCORE}")
                rally_ended = True
                break
            time.sleep(0 if LEARNING_MODE else 1.0 / CFG.ANIMATION_FPS)
        
        # Handle paddle hit from second parabola
        if rally_ended or paddle_hit:
            if paddle_hit:
                try:
                    paddle_pos = pyb.getBasePositionAndOrientation(
                        _PADDLE1.body_id if last_hitter == 'paddle1' else _PADDLE2.body_id
                    )[0]
                    if last_hitter == 'paddle1':
                        import player1
                        strength, angle_deg, spin_deg = player1.get_shot_params(paddle_pos[0], paddle_pos[1])
                    else:
                        import player2
                        strength, angle_deg, spin_deg = player2.get_shot_params(paddle_pos[0], paddle_pos[1])
                except Exception:
                    strength, angle_deg, spin_deg = base_strength, base_angle, base_spin
                
                temp_params = {'strength': strength, 'angle_deg': angle_deg, 'spin_deg': spin_deg}
                try:
                    pyb.resetBasePositionAndOrientation(_BALL.body_id, hit_pos, pyb.getQuaternionFromEuler([0, 0, 0]))
                except Exception:
                    pass
                continue
            break
        
        # ========== END OF RALLY ==========
        # Last hitter scores and records learning
        ball_on_opponent_side = (last_hitter == 'paddle1' and ball_pos[0] > 0) or \
                                (last_hitter == 'paddle2' and ball_pos[0] < 0)
        
        SCORE[last_hitter] += 1
        print(f"[SCORE] {last_hitter} +1 (end of second) → {SCORE}")
        
        # Record successful shot in learning system (attempt regardless of
        # ball_on_opponent_side; player.record_success will no-op if missing
        # data). This makes learning more likely during headless runs so CSVs
        # are populated for inspection and diffusion tests.
        try:
            paddle_pos = pyb.getBasePositionAndOrientation(
                _PADDLE1.body_id if last_hitter == 'paddle1' else _PADDLE2.body_id
            )[0]
            if last_hitter == 'paddle1':
                import player1
                player1.record_success()
            else:
                import player2
                player2.record_success()
            print(f"[LEARN] Recorded success for {last_hitter} at ({paddle_pos[0]:.3f}, {paddle_pos[1]:.3f})")
        except Exception:
            pass
        
        rally_ended = True
    
    # ========== POST-RALLY ==========
    _NEXT_SERVER = 2 if _NEXT_SERVER == 1 else 1
    print(f">>> Rally ended. Score: {SCORE}")
    
    # Check if match is over
    if WINNING_SCORE > 0:
        if SCORE['paddle1'] >= WINNING_SCORE:
            print(f"\n{'='*50}")
            print(f"🎉 MATCH OVER: paddle1 wins {SCORE['paddle1']}-{SCORE['paddle2']}!")
            print(f"{'='*50}\n")
            return False
        elif SCORE['paddle2'] >= WINNING_SCORE:
            print(f"\n{'='*50}")
            print(f"🎉 MATCH OVER: paddle2 wins {SCORE['paddle2']}-{SCORE['paddle1']}!")
            print(f"{'='*50}\n")
            return False
    
    return True


def series_rallies(count: int = 5, delay: float = 0.5) -> bool:
    """Run multiple rallies in sequence."""
    if _PADDLE1 is None:
        start()
    
    for i in range(count):
        print(f"\n{'='*50}")
        print(f"Rally {i+1}/{count}")
        print(f"{'='*50}")
        try:
            rally()
        except Exception as e:
            print(f"Rally aborted: {e}")
            break
        time.sleep(delay)
    
    print(f"\nFinal score: {SCORE}")
    return True

# ============================================================================
# INPUT POLLING (Keyboard Controls)
# ============================================================================

def _input_thread_loop(stop_event: threading.Event, poll_interval: float = 0.05):
    """Background thread polling player controllers."""
    while not stop_event.is_set():
        try:
            try:
                import player1
                player1.update_from_keyboard()
            except Exception:
                pass
            try:
                import player2
                player2.update_from_keyboard()
            except Exception:
                pass
        except Exception:
            pass
        time.sleep(poll_interval)


def _start_input_thread():
    """Start keyboard polling thread."""
    global _INPUT_THREAD, _INPUT_THREAD_STOP
    if _INPUT_THREAD and _INPUT_THREAD.is_alive():
        return
    _INPUT_THREAD_STOP = threading.Event()
    _INPUT_THREAD = threading.Thread(target=_input_thread_loop, args=(_INPUT_THREAD_STOP,), daemon=True)
    _INPUT_THREAD.start()


def _stop_input_thread(timeout: float = 1.0):
    """Stop keyboard polling thread."""
    global _INPUT_THREAD, _INPUT_THREAD_STOP
    if _INPUT_THREAD_STOP is None:
        return
    try:
        _INPUT_THREAD_STOP.set()
        if _INPUT_THREAD:
            _INPUT_THREAD.join(timeout)
    finally:
        _INPUT_THREAD = None
        _INPUT_THREAD_STOP = None

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("Ping-Pong Simulator")
    print("=" * 50)
    try:
        start()
        print("\nQuick start:")
        print("  environment.rally()          # Run 1 rally")
        print("  environment.series_rallies(5)  # Run 5 rallies")
        print("\nPress Ctrl+C to exit")
        
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop()
