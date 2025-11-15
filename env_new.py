"""
Lightweight extraction of Table, Paddle and Ball from environment.py.

This module provides simple object classes that create the visual/collision
geometry in PyBullet and a small helper to decide which axis will be
considered parallel to the net (based on table dimensions).

Designed as the starting point for moving the original, larger
`environment.py` into a clearer, object-oriented structure.
"""
from dataclasses import dataclass
import logging
import time
import random
import numpy as np
import pybullet as pyb
import pybullet_data
import threading

logger = logging.getLogger("env_new")
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class Config:
    # Table dimensions (ITTF standard)
    TABLE_L: float = 2.74  # length (player-to-player)
    TABLE_W: float = 1.525 # width (left-right)
    TABLE_H: float = 0.76  # table surface height
    TABLE_THICKNESS: float = 0.05
    NET_HEIGHT: float = 0.1525
    NET_THICKNESS: float = 0.02

    # Ball
    BALL_RADIUS: float = 0.02

    # Paddle
    BLADE_LENGTH: float = 0.13
    PADDLE_WIDTH: float = 0.15
    PADDLE_HEIGHT: float = 0.005
    HANDLE_LENGTH: float = 0.06


CFG = Config()

# Module-level environment state
_CLIENT = None
_SC = None
_TABLE = None
_PADDLE1 = None
_PADDLE2 = None
_BALL = None
_NEXT_SERVER = 1  # 1 or 2, alternates each rally

# Input thread for polling keyboard controllers
_INPUT_THREAD = None
_INPUT_THREAD_STOP = None

# default player shot params (can be adjusted or replaced by user code)
PLAYER_PARAMS = {
    'paddle1': {'strength': 1.0, 'angle_deg': 0.0, 'spin_deg': 0.0},
    'paddle2': {'strength': 1.0, 'angle_deg': 0.0, 'spin_deg': 0.0},
}

# Score tracking
SCORE = {'paddle1': 0, 'paddle2': 0}

# module-level debug ids for drawn lines
debug_ids = []
# collision debounce state shared across animation calls (module-level)
active_collisions = {"table": False, "net": False, "paddle1": False, "paddle2": False}

# Noise sensitivity constants: how much the distance between ball and optimal hit
# contributes to perturbations in strength and angle
NOISE_STRENGTH_SENSITIVITY = 0.0  # multiplied by distance to add to strength
NOISE_ANGLE_SENSITIVITY = 0.0    # multiplied by distance to add to angle (degrees)


def clear_debug():
    """Remove any user-debug items recorded in `debug_ids`."""
    global debug_ids
    for did in list(debug_ids):
        try:
            pyb.removeUserDebugItem(did)
        except Exception:
            pass
    debug_ids = []


def _distance_xy(a, b):
    return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))


def get_optimal_hit_point(paddle_obj):
    """Return the optimal hit point (middle of blade, BALL_RADIUS closer to net).

    This is the module-level version used by top-level rally/control functions.
    """
    pos, ori = pyb.getBasePositionAndOrientation(paddle_obj.body_id)
    pos = np.array(pos)
    rot = np.array(pyb.getMatrixFromQuaternion(ori)).reshape(3, 3)
    forward = rot.dot(np.array([1.0, 0.0, 0.0]))
    optimal = pos + forward * CFG.BALL_RADIUS
    return tuple(optimal.tolist())


def create_ball_at(position):
    global _BALL, _SC
    if _BALL is not None:
        try:
            pyb.removeBody(_BALL.body_id)
        except Exception:
            pass
    _BALL = Ball(_SC, start_pos=list(position))
    _BALL.create()
    return _BALL


def generate_shot_parabolas(ball_pos, strength=1.0, angle_deg=0.0, spin_deg=0.0, samples=128):
    """Generate two successive parabolic trajectories from current ball position.

    The implementation is intentionally simple and deterministic: it builds a
    parabola in the local (forward) direction with a vertex height chosen to
    be above the net. The second parabola is produced by starting at the
    landing/bounce point of the first and applying an additional rotation
    given by `spin_deg`.

    Returns: pts1, pts2, meta where each pts* is a list of (x,y,z) points and
    meta contains a small dict with keys 'bounce1' and 'bounce2'.
    """
    c = CFG
    bx, by, bz = ball_pos

    angle_rad = float(np.radians(angle_deg))
    spin_rad = float(np.radians(spin_deg))

    table_z = c.TABLE_H + c.BALL_RADIUS

    # choose horizontal travel distance proportional to strength
    max_d = 0.75 * c.TABLE_L
    d = max_d * float(strength)

    # choose direction sign based on ball x (send towards opponent)
    dir_sign = 1 if bx <= 0 else -1

    # origin: slightly behind the contact point at table height (not ball z).
    # We'll compute the parabola at table height, then shift it in XY so the
    # point on the parabola with the same Z as the ball lines up with the
    # ball's XY position. This preserves continuity of the trajectory.
    origin = (bx - 0.02 * dir_sign, by, table_z)

    pts1, left1, right1 = generate_parabola_points(origin, contact_point=ball_pos, power=d, angle_rad=angle_rad, anchor_point=ball_pos, samples=samples, dir_sign=dir_sign)

    # second parabola: start at landing (use right1 as bounce point), add spin
    bounce = right1
    bounce_anchor = (bounce[0], bounce[1], table_z)
    pts2, left2, right2 = generate_parabola_points(bounce_anchor, contact_point=bounce_anchor, power=d, angle_rad=angle_rad + spin_rad, anchor_point=bounce_anchor, samples=samples, dir_sign=dir_sign)

    meta = {'bounce1': right1, 'bounce2': right2}
    return pts1, pts2, meta


def generate_parabola_points(origin, contact_point, power=0.5, angle_rad=0.0, anchor_point=None, samples=128, dir_sign=1):
    """Create a single parabola period as a list of world-space (x,y,z) points.

    origin: start position (x,y,z) at or near table height
    contact_point: the point where the ball was hit (used to compute starting z)
    power: horizontal travel distance to landing point
    angle_rad: rotation to apply to the local-forward axis
    anchor_point: rotation anchor (world x,y) — defaults to contact_point
    dir_sign: +1 forwards in +local-x, -1 for opposite
    Returns: points, left_contact, right_contact (both contact tuples at landing)
    """
    c = CFG
    if anchor_point is None:
        anchor_point = (contact_point[0], contact_point[1])
    ax, ay = anchor_point[0], anchor_point[1]

    table_z = c.TABLE_H + c.BALL_RADIUS

    # horizontal distance to land
    D = float(power)
    if D <= 1e-6:
        D = 0.1

    # vertex (apex) position along local x (midpoint)
    h = D / 2.0
    k = c.TABLE_H + (4.0 / 3.0) * c.NET_HEIGHT

    # compute 'a' so parabola passes through origin.z at local x=0 and vertex at (h,k)
    z0 = float(origin[2])
    denom = (0 - h) ** 2
    a = (z0 - k) / denom if denom != 0 else -1e-6

    points = []
    for i in range(samples):
        s = (i / float(samples - 1)) * D
        lx = dir_sign * s
        z = a * (s - h) ** 2 + k

        # rotate local (lx,0) by angle_rad around anchor (ax,ay)
        dx = lx
        dy = 0.0
        wx = ax + dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
        wy = ay + dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
        points.append((wx, wy, float(z)))

    # Optionally align parabola in XY so a point on the parabola that has
    # the same upward-axis coordinate as `contact_point` matches the contact
    # point's XY. This avoids teleporting the ball while preserving the
    # original parabola shape.
    try:
        contact_z = float(contact_point[2])
        # find index with minimal abs(z - contact_z)
        best_idx = 0
        best_diff = abs(points[0][2] - contact_z)
        for i, p in enumerate(points):
            d = abs(p[2] - contact_z)
            if d < best_diff:
                best_diff = d
                best_idx = i
        # compute XY offset required to bring that parabola point to contact_point
        px, py, pz = points[best_idx]
        dx_off = float(contact_point[0]) - px
        dy_off = float(contact_point[1]) - py
        if abs(dx_off) > 1e-9 or abs(dy_off) > 1e-9:
            points = [(x + dx_off, y + dy_off, z) for (x, y, z) in points]
    except Exception:
        # if contact_point not provided or malformed, skip alignment
        pass

    # landing point is last point
    landing = points[-1]
    # produce simple left/right contact variants (small lateral offsets)
    left_contact = (landing[0] - 0.01, landing[1], table_z)
    right_contact = (landing[0] + 0.01, landing[1], table_z)

    return points, left_contact, right_contact




class ShapeCache:
    """Very small cache to avoid creating duplicate visual/collision shapes."""
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
    """Creates a table body, decorative blue squares and legs, plus a net."""
    def __init__(self, shape_cache: ShapeCache, config: Config = CFG, base_pos=(0, 0, 0)):
        self.sc = shape_cache
        self.c = config
        self.base_pos = np.array(base_pos)
        self.table_id = None
        self.net_id = None
        self.legs = []
        self.decor_ids = []

    def create(self):
        c = self.c
        half_extents = [c.TABLE_L / 2, c.TABLE_W / 2, c.TABLE_THICKNESS / 2]
        col = self.sc.box_collision(half_extents)
        vis = self.sc.box_visual(half_extents, [1, 1, 1, 1])

        table_z = c.TABLE_H - c.TABLE_THICKNESS / 2
        self.table_id = pyb.createMultiBody(baseMass=0,
                                             baseCollisionShapeIndex=col,
                                             baseVisualShapeIndex=vis,
                                             basePosition=[0, 0, table_z])

        # decorative blue squares on top (visual only)
        square_half_L = 0.65
        square_half_W = 0.3475
        square_half_H = 0.001
        square_vis = self.sc.box_visual([square_half_L, square_half_W, square_half_H], [0.05, 0.15, 0.45, 1.0])
        # place four small visuals near the corners on top
        offs_x = (c.TABLE_L / 2) - square_half_L - 0.05
        offs_y = (c.TABLE_W / 2) - square_half_W - 0.05
        square_z = c.TABLE_H + square_half_H
        for sx in (-offs_x, offs_x):
            for sy in (-offs_y, offs_y):
                vis_id = pyb.createMultiBody(baseMass=0, baseVisualShapeIndex=square_vis, basePosition=[sx, sy, square_z])
                self.decor_ids.append(vis_id)

        # simple legs (visual + collision)
        leg_half = [0.02, 0.02, 0.4]
        leg_col = self.sc.box_collision(leg_half)
        leg_vis = self.sc.box_visual(leg_half, [0.1, 0.1, 0.1, 1.0])
        leg_x = (c.TABLE_L / 2) - 0.1
        leg_y = (c.TABLE_W / 2) - 0.1
        leg_z = c.TABLE_H - c.TABLE_THICKNESS / 2 - leg_half[2]
        for sx in (-leg_x, leg_x):
            for sy in (-leg_y, leg_y):
                leg_id = pyb.createMultiBody(baseMass=0, baseCollisionShapeIndex=leg_col, baseVisualShapeIndex=leg_vis, basePosition=[sx, sy, leg_z])
                self.legs.append(leg_id)

        # net (visual + collision)
        net_half = [c.NET_THICKNESS / 2, c.TABLE_W / 2, c.NET_HEIGHT / 2]
        net_col = self.sc.box_collision(net_half)
        net_vis = self.sc.box_visual(net_half, [0.0, 0.0, 0.0, 1.0])
        # place net centered on x=0, top of net sits on table surface
        net_z = c.TABLE_H + net_half[2]
        self.net_id = pyb.createMultiBody(baseMass=0, baseCollisionShapeIndex=net_col, baseVisualShapeIndex=net_vis, basePosition=[0, 0, net_z])

        return self.table_id


class Paddle:
    """Create a simple paddle made from two boxes (blade + handle)."""
    def __init__(self, shape_cache: ShapeCache, config: Config = CFG, base_pos=(0, 0, 0), base_yaw=1.57079633, color=(1, 0, 0, 1)):
        self.sc = shape_cache
        self.c = config
        self.base_pos = list(base_pos)
        self.base_yaw = base_yaw
        self.color = list(color)
        self.body_id = None

    def create(self):
        c = self.c
        blade_half = [c.BLADE_LENGTH / 2, c.PADDLE_WIDTH / 2, c.PADDLE_HEIGHT / 2]
        blade_col = self.sc.box_collision(blade_half)
        blade_vis = self.sc.box_visual(blade_half, self.color)

        handle_half = [c.HANDLE_LENGTH / 2, 0.015 / 2, c.PADDLE_HEIGHT / 2]
        handle_vis = self.sc.box_visual(handle_half, [0.1, 0.1, 0.1, 1.0])

        # create base body (blade) and attach a fixed handle link so it visually sticks out
        linkMasses = [0.0]
        linkCollisionShapes = [-1]
        linkVisualShapes = [handle_vis]
        handle_offset_y = c.PADDLE_WIDTH / 2 + handle_half[0]
        linkPositions = [[0, handle_offset_y, 0]]
        linkOrientations = [pyb.getQuaternionFromEuler([0, 0, np.radians(90)])]
        linkInertialFramePositions = [[0, 0, 0]]
        linkInertialFrameOrientations = [[0, 0, 0, 1]]
        linkParentIndices = [0]
        linkJointTypes = [pyb.JOINT_FIXED]
        linkJointAxis = [[0, 0, 0]]

        base_orientation = pyb.getQuaternionFromEuler([np.radians(90), np.radians(90), self.base_yaw])

        self.body_id = pyb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=blade_col,
            baseVisualShapeIndex=blade_vis,
            basePosition=self.base_pos,
            baseOrientation=base_orientation,
            linkMasses=linkMasses,
            linkCollisionShapeIndices=linkCollisionShapes,
            linkVisualShapeIndices=linkVisualShapes,
            linkPositions=linkPositions,
            linkOrientations=linkOrientations,
            linkInertialFramePositions=linkInertialFramePositions,
            linkInertialFrameOrientations=linkInertialFrameOrientations,
            linkParentIndices=linkParentIndices,
            linkJointTypes=linkJointTypes,
            linkJointAxis=linkJointAxis
        )

        return self.body_id


class Ball:
    """Create a simple ball object (visual + collision)."""
    def __init__(self, shape_cache: ShapeCache, config: Config = CFG, start_pos=(0, 0, 1.0)):
        self.sc = shape_cache
        self.c = config
        self.start_pos = list(start_pos)
        self.body_id = None

    def create(self):
        col = self.sc.sphere_collision(self.c.BALL_RADIUS)
        vis = self.sc.sphere_visual(self.c.BALL_RADIUS, [1, 1, 1, 1])
        self.body_id = pyb.createMultiBody(baseMass=0.0027, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=self.start_pos)
        return self.body_id


def choose_net_axis(config: Config = CFG):
    """Decide which world axis should be considered parallel to the net.

    Rule used here:
    - Compare table length and table width. The axis corresponding to the
      larger measurement will be treated as the table's long axis.
    - The net is placed perpendicular to the player-to-player axis, so it
      runs parallel to the table's short axis in a real table. To keep
      things explicit we return both the long axis and the net axis.
    Returns: dict with keys `long_axis` and `net_axis` containing one of
    'X','Y','Z' (Z won't be chosen, but kept for clarity).
    """
    if config.TABLE_L >= config.TABLE_W:
        long_axis = 'X'  # length runs along X (player-to-player)
        net_axis = 'Y'   # net runs along Y (left-right)
    else:
        long_axis = 'Y'
        net_axis = 'X'
    logger.debug("choose_net_axis -> long_axis=%s net_axis=%s", long_axis, net_axis)
    return {'long_axis': long_axis, 'net_axis': net_axis}


def demo(start_gui=False):
    """Lightweight demo wrapper: starts the environment and returns True.

    The heavy demo implementation was removed to avoid duplicated helpers
    and nested scopes. Use `start_env()` + `rally()` for interactive testing.
    """
    return start_env(start_gui=start_gui)

    # legacy demo-local collision helpers removed; use module-level versions instead


def rally():
    """Start and run a rally from the server indicated by `_NEXT_SERVER`.

    This function creates the ball at the optimal hit point of the serving
    paddle, then simulates the rally following the rules provided by the user.
    It alternates `_NEXT_SERVER` each time it finishes.
    """
    global _NEXT_SERVER, _BALL, _PADDLE1, _PADDLE2
    if _PADDLE1 is None or _PADDLE2 is None or _SC is None:
        raise RuntimeError("Environment not started — call start_env() first")

    server = 'paddle1' if _NEXT_SERVER == 1 else 'paddle2'
    server_obj = _PADDLE1 if server == 'paddle1' else _PADDLE2

    # Reset player controllers (if present) at start of rally so controls
    # and parameters return to defaults.
    try:
        import player1 as p1ctrl
        p1ctrl.reset_for_new_rally()
    except Exception:
        pass
    try:
        import player2 as p2ctrl
        p2ctrl.reset_for_new_rally()
    except Exception:
        pass

    # place ball at optimal hit point of server
    start_pt = get_optimal_hit_point(server_obj)
    _BALL = create_ball_at(start_pt)

    # rally state
    last_hitter = server  # server just 'hit' the ball
    rally_ended = False

    # main rally loop
    current_hitter = server
    params = PLAYER_PARAMS[server]

    while not rally_ended:
        # current ball position
        ball_pos = pyb.getBasePositionAndOrientation(_BALL.body_id)[0]

        # Always apply noise: compute distance from ball to optimal hit point
        paddle_obj = _PADDLE1 if current_hitter == 'paddle1' else _PADDLE2
        optimal = get_optimal_hit_point(paddle_obj)
        dist = _distance_xy(ball_pos, optimal)
        noise = dist * random.uniform(-1.0, 1.0)

        # Get base parameters and clamp to valid ranges BEFORE adding noise
        base_strength = float(np.clip(params.get('strength', 1.0), 0.4, 1.6))
        base_angle = float(np.clip(params.get('angle_deg', 0.0), -33.0, 33.0))
        base_spin = float(np.clip(params.get('spin_deg', 0.0), -12.0, 12.0))

        # Apply noise: add to clamped base values (noise can extend beyond ranges)
        strength = base_strength + noise * NOISE_STRENGTH_SENSITIVITY
        angle_deg = base_angle + noise * NOISE_ANGLE_SENSITIVITY
        spin_deg = base_spin

        # generate two parabolas from current ball position using current params with noise
        pts1, pts2, meta = generate_shot_parabolas(ball_pos, strength=strength, angle_deg=angle_deg, spin_deg=spin_deg)

        # helper paddle positions
        p1_pos = np.array(pyb.getBasePositionAndOrientation(_PADDLE1.body_id)[0])
        p2_pos = np.array(pyb.getBasePositionAndOrientation(_PADDLE2.body_id)[0])

        # find nearest start index in pts1 relative to ball XY
        start_idx = 0
        min_d = float('inf')
        for i, p in enumerate(pts1):
            d = np.linalg.norm(np.array(p[:2]) - np.array(ball_pos[:2]))
            if d < min_d:
                min_d = d
                start_idx = i

        # play along first parabola
        paddle_hit = False
        for p in pts1[start_idx:]:
            try:
                pyb.resetBasePositionAndOrientation(_BALL.body_id, p, pyb.getQuaternionFromEuler([0, 0, 0]))
            except Exception:
                pass
            try:
                pyb.stepSimulation()
            except Exception:
                rally_ended = True
                break

            # net collision -> rally ends; non-last-hitter scores
            if check_collision_with_net(p):
                # award point to the player who did NOT last hit the ball
                if last_hitter == 'paddle1':
                    SCORE['paddle2'] += 1
                    print("Score: paddle2 +1 (net).", SCORE)
                else:
                    SCORE['paddle1'] += 1
                    print("Score: paddle1 +1 (net).", SCORE)
                rally_ended = True
                break

            # paddle collisions: if a different paddle than last_hitter hits, handle hit
            if check_collision_with_paddle(p, p1_pos) and last_hitter != 'paddle1':
                # paddle1 hit: rally continues but trajectories are discarded
                last_hitter = 'paddle1'
                current_hitter = 'paddle1'
                paddle_hit = True
                hit_pos = p
                break
            if check_collision_with_paddle(p, p2_pos) and last_hitter != 'paddle2':
                last_hitter = 'paddle2'
                current_hitter = 'paddle2'
                paddle_hit = True
                hit_pos = p
                break

            # record table collisions but do not interrupt
            _ = check_collision_with_table(p)

            time.sleep(1.0/240.0)

        if rally_ended:
            break

        if paddle_hit:
            # When paddle is hit: stop animating, discard current parabolas,
            # compute new parabolas now applying noise and alignment, then continue
            # the loop which will animate the newly computed trajectories.
            # compute noise based on distance to optimal hit of the hitter
            paddle_obj = _PADDLE1 if current_hitter == 'paddle1' else _PADDLE2
            optimal = get_optimal_hit_point(paddle_obj)
            dist = _distance_xy(hit_pos, optimal)
            noise = dist * random.uniform(-1.0, 1.0)

            # Get base parameters and clamp to valid ranges BEFORE adding noise
            base_strength = float(np.clip(PLAYER_PARAMS[current_hitter].get('strength', 1.0), 0.4, 1.6))
            base_angle = float(np.clip(PLAYER_PARAMS[current_hitter].get('angle_deg', 0.0), -33.0, 33.0))
            base_spin = float(np.clip(PLAYER_PARAMS[current_hitter].get('spin_deg', 0.0), -12.0, 12.0))

            # Apply noise: add to clamped base values (noise can extend beyond ranges)
            strength = base_strength + noise * NOISE_STRENGTH_SENSITIVITY
            angle_deg = base_angle + noise * NOISE_ANGLE_SENSITIVITY
            spin_deg = base_spin

            # generate new parabolas from the hit position; pass contact_point=hit_pos
            pts1, pts2, meta = generate_shot_parabolas(hit_pos, strength=strength, angle_deg=angle_deg, spin_deg=spin_deg)
            # loop continues to animate the newly computed pts1 (start of while)
            continue

        # If we reach here, first parabola finished without a paddle hit.
        # According to spec, if the second parabola does not hit the table at any point,
        # the rally ends when the ball is at the beginning of the second parabola.
        will_hit_table = False
        for p in pts2:
            if check_collision_with_table(p):
                will_hit_table = True
                break

        if not will_hit_table:
            # End rally: second parabola will not hit the table -> non-last-hitter scores
            print("Rally end: second parabola will not hit the table")
            if last_hitter == 'paddle1':
                SCORE['paddle2'] += 1
                print("Score: paddle2 +1 (no-table second).", SCORE)
            else:
                SCORE['paddle1'] += 1
                print("Score: paddle1 +1 (no-table second).", SCORE)
            rally_ended = True
            break

        # Animate second parabola fully; if paddle hits occur, handle similarly
        table_hit_in_second = False
        for p in pts2:
            try:
                pyb.resetBasePositionAndOrientation(_BALL.body_id, p, pyb.getQuaternionFromEuler([0, 0, 0]))
            except Exception:
                pass
            try:
                pyb.stepSimulation()
            except Exception:
                rally_ended = True
                break

            # net collision during second parabola -> non-last-hitter scores
            if check_collision_with_net(p):
                if last_hitter == 'paddle1':
                    SCORE['paddle2'] += 1
                    print("Score: paddle2 +1 (net).", SCORE)
                else:
                    SCORE['paddle1'] += 1
                    print("Score: paddle1 +1 (net).", SCORE)
                rally_ended = True
                break

            if check_collision_with_table(p):
                table_hit_in_second = True

            # paddle checks
            if check_collision_with_paddle(p, p1_pos) and last_hitter != 'paddle1':
                last_hitter = 'paddle1'
                current_hitter = 'paddle1'
                # compute new trajectories from this hit (noise applied)
                hit_pos = p
                paddle_hit = True
                break
            if check_collision_with_paddle(p, p2_pos) and last_hitter != 'paddle2':
                last_hitter = 'paddle2'
                current_hitter = 'paddle2'
                hit_pos = p
                paddle_hit = True
                break

            time.sleep(1.0/240.0)

        if rally_ended:
            break

        if paddle_hit:
            # compute noise and regenerate parabolas from hit_pos
            paddle_obj = _PADDLE1 if current_hitter == 'paddle1' else _PADDLE2
            optimal = get_optimal_hit_point(paddle_obj)
            dist = _distance_xy(hit_pos, optimal)
            noise = dist * random.uniform(-1.0, 1.0)

            # Get base parameters and clamp to valid ranges BEFORE adding noise
            base_strength = float(np.clip(PLAYER_PARAMS[current_hitter].get('strength', 1.0), 0.4, 1.6))
            base_angle = float(np.clip(PLAYER_PARAMS[current_hitter].get('angle_deg', 0.0), -33.0, 33.0))
            base_spin = float(np.clip(PLAYER_PARAMS[current_hitter].get('spin_deg', 0.0), -12.0, 12.0))

            # Apply noise: add to clamped base values (noise can extend beyond ranges)
            strength = base_strength + noise * NOISE_STRENGTH_SENSITIVITY
            angle_deg = base_angle + noise * NOISE_ANGLE_SENSITIVITY
            spin_deg = base_spin

            pts1, pts2, meta = generate_shot_parabolas(hit_pos, strength=strength, angle_deg=angle_deg, spin_deg=spin_deg)
            continue

        # If we reached the end of pts2 and it hit the table, rally ends per spec
        if table_hit_in_second:
            # point to the player who last hit the ball
            if last_hitter == 'paddle1':
                SCORE['paddle1'] += 1
                print("Score: paddle1 +1 (end of second).", SCORE)
            else:
                SCORE['paddle2'] += 1
                print("Score: paddle2 +1 (end of second).", SCORE)
            rally_ended = True
            break

        # default fallback end
        rally_ended = True

    # alternate server for next rally
    _NEXT_SERVER = 2 if _NEXT_SERVER == 1 else 1
    return True


def series_rallies(count=10, start_gui=False, delay=0.5):
    """Run `rally()` `count` times in sequence, alternating server each rally.

    If the environment is not started this will call `start_env()` with
    `start_gui` to initialize. Sleeps `delay` seconds between rallies.
    Returns True on normal completion.
    """
    if _PADDLE1 is None or _PADDLE2 is None or _SC is None:
        start_env(start_gui=start_gui)

    completed = 0
    try:
        for i in range(count):
            srv = 'paddle1' if _NEXT_SERVER == 1 else 'paddle2'
            print(f"Starting rally {i+1}/{count} (server={srv})")
            try:
                rally()
            except Exception as e:
                print(f"Rally {i+1} aborted with exception: {e}")
                break
            completed += 1
            time.sleep(delay)
    except KeyboardInterrupt:
        print("Series interrupted by user")

    print(f"Series finished: {completed}/{count} rallies completed")
    return True


def check_collision_with_table(ball_pos, c=CFG):
    """Check if ball position is at table surface (within BALL_RADIUS).
    Returns True if collision detected."""
    table_surface_z = c.TABLE_H + c.BALL_RADIUS
    ball_z = ball_pos[2]
    hit = abs(ball_z - table_surface_z) < 0.001  # small epsilon for floating point
    if hit:
        print(f"Collision detected: table at pos {tuple(ball_pos)}")
    return hit


def check_collision_with_net(ball_pos, c=CFG):
    """Check if ball hits the net (thin box centered at x=0).
    Returns True if collision detected."""
    # net is at x=0, extends from -TABLE_W/2 to +TABLE_W/2 in Y, up to NET_HEIGHT
    net_x_range = c.NET_THICKNESS / 2
    net_y_range = c.TABLE_W / 2
    net_top_z = c.TABLE_H + c.NET_HEIGHT

    ball_x, ball_y, ball_z = ball_pos
    # collision if ball is within net bounds
    within_x = abs(ball_x) < net_x_range + c.BALL_RADIUS
    within_y = abs(ball_y) < net_y_range + c.BALL_RADIUS
    within_z = ball_z < net_top_z + c.BALL_RADIUS
    hit = within_x and within_y and within_z
    if hit:
        print(f"Collision detected: net at pos {tuple(ball_pos)}")
    return hit


def check_collision_with_paddle(ball_pos, paddle_pos, c=CFG):
    """Check if ball collides with paddle head (only the blade, not handle).
    Returns True if collision detected."""
    blade_half_x = c.PADDLE_HEIGHT / 2
    blade_half_y = c.PADDLE_WIDTH / 2
    blade_half_z = c.BLADE_LENGTH / 2

    paddle_x, paddle_y, paddle_z = paddle_pos
    ball_x, ball_y, ball_z = ball_pos

    # AABB test with padding for ball radius
    dx = abs(ball_x - paddle_x)
    dy = abs(ball_y - paddle_y)
    dz = abs(ball_z - paddle_z)
    hit = (dx < blade_half_x + c.BALL_RADIUS and
            dy < blade_half_y + c.BALL_RADIUS and
            dz < blade_half_z + c.BALL_RADIUS)
    if hit:
        print(f"Collision detected: paddle at pos {tuple(ball_pos)} (paddle_pos={tuple(paddle_pos)})")
    return hit


def draw_and_animate_trajectory(contact_point, power=1.0, angle_1=0.0, spin=0.0, animate=True, anchor_override=None, dir_sign=1):
    """Generate, draw, and optionally animate the full two-period trajectory.

    Args:
        contact_point: (x, y, z) where the ball is hit
        power: shot strength (affects horizontal distance)
        angle_1: rotation angle (radians) for first parabola around contact point
        spin: spin value that rotates second parabola (additional angle applied at bounce)
        animate: if True, animate the ball along the trajectory
        dir_sign: +1 for shots towards +X, -1 for shots towards -X
    """
    clear_debug()

    # First period: from contact point with angle_1, anchor at contact_point (or override)
    origin = (contact_point[0] - 0.02, contact_point[1], CFG.TABLE_H + CFG.BALL_RADIUS)
    anchor1 = contact_point if anchor_override is None else anchor_override
    pts1, left1, right1 = generate_parabola_points(origin, contact_point, power=power, angle_rad=angle_1, anchor_point=anchor1, samples=128, dir_sign=dir_sign)
    bounce1 = right1
    draw_parabola(pts1, color=(1, 1, 0))

    # Second period: from bounce1 with spin applied
    # spin adds to angle for the second parabola, anchor at bounce1 (start of second parabola)
    angle_2 = angle_1 + spin
    anchor2 = bounce1 if anchor_override is None else anchor_override
    pts2, left2, right2 = generate_parabola_points(bounce1, bounce1, power=power, angle_rad=angle_2, anchor_point=anchor2, samples=128, dir_sign=dir_sign)
    bounce2 = right2
    draw_parabola(pts2, color=(0.2, 1.0, 0.2))

    print(f"Trajectory: contact={contact_point}, power={power}, angle_1={np.degrees(angle_1):.1f}°, spin={np.degrees(spin):.1f}°")
    print(f"  bounce1={bounce1}, bounce2={bounce2}")

    # Find nearest start point within the first period and produce a combined sequence
    # If dir_sign == 1 we play points in forward index order; if dir_sign == -1 we reverse
    start_idx = 0
    min_dist = float('inf')
    for idx, p in enumerate(pts1):
        dist = np.sqrt((p[0] - contact_point[0])**2 + (p[1] - contact_point[1])**2 + (p[2] - contact_point[2])**2)
        if dist < min_dist:
            min_dist = dist
            start_idx = idx

    if dir_sign == 1:
        combined = pts1[start_idx:] + pts2
    else:
        rev1 = list(reversed(pts1))
        rev2 = list(reversed(pts2))
        # find nearest index in reversed list
        start_idx_rev = 0
        min_dist = float('inf')
        for idx, p in enumerate(rev1):
            dist = np.sqrt((p[0] - contact_point[0])**2 + (p[1] - contact_point[1])**2 + (p[2] - contact_point[2])**2)
            if dist < min_dist:
                min_dist = dist
                start_idx_rev = idx
        combined = rev1[start_idx_rev:] + rev2

    if not animate:
        return combined, bounce1, bounce2

    # Animation loop with collision handling
    while True:
        # ensure a ball exists
        if _BALL is None:
            create_ball_at(combined[0])

        events = move_ball_along(_BALL.body_id, combined, step_time=1.0/240.0)
        # events is a list of (obj_name, idx, pos)
        if not events:
            # no collisions, loop and replay trajectory
            continue

        # handle first significant event per returned list
        ev = events[0]
        obj, idx, pos = ev
        if obj.startswith("paddle"):
            which = obj  # 'paddle1' or 'paddle2'
            paddle_obj = _PADDLE1 if which == 'paddle1' else _PADDLE2
            paddle_pos, paddle_ori = pyb.getBasePositionAndOrientation(paddle_obj.body_id)
            paddle_pos = np.array(paddle_pos)
            rot = np.array(pyb.getMatrixFromQuaternion(paddle_ori)).reshape(3, 3)
            forward = rot.dot(np.array([1.0, 0.0, 0.0]))

            # optimal hit point: middle of paddle head, ball radius in front
            optimal_hit = paddle_pos + forward * (CFG.BALL_RADIUS + 0.0)

            # current ball position
            ball_pos = np.array(pos)
            dist = np.linalg.norm(ball_pos - optimal_hit)

            # compute noise
            noise_factor = random.uniform(-0.5, 0.5)
            strength = float(np.clip(1.0 + dist * noise_factor, 0.4, 1.6))
            angle_noise_deg = dist * noise_factor * 20.0
            angle_deg = float(np.clip(angle_noise_deg, -33.0, 33.0))
            angle_rad = np.radians(angle_deg)

            anchor = tuple(ball_pos.tolist())

            outgoing_contact = tuple(ball_pos.tolist())
            if which == 'paddle1':
                combined, bounce1, bounce2 = draw_and_animate_trajectory(outgoing_contact, power=strength, angle_1=angle_rad, spin=0.0, animate=False, anchor_override=anchor, dir_sign=1)
            else:
                combined, bounce1, bounce2 = draw_and_animate_trajectory(outgoing_contact, power=strength, angle_1=-angle_rad, spin=0.0, animate=False, anchor_override=anchor, dir_sign=-1)

            # loop to animate the newly created trajectory
            continue

        # other collisions (table/net) -> just print and continue
        print(f"Collision event: {ev}")
        continue
def draw_parabola(points, color=(1, 1, 0), line_width=2):
    global debug_ids
    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        try:
            did = pyb.addUserDebugLine(a, b, lineColorRGB=color, lineWidth=line_width, lifeTime=0)
            debug_ids.append(did)
        except Exception:
            pass


def move_ball_along(body_id, points, step_time=1.0/240.0):
    """Simple animator: teleport ball to each point and step simulation.
    Also checks for collisions with table, net, and paddles."""
    paddle1_pos = np.array(pyb.getBasePositionAndOrientation(_PADDLE1.body_id)[0]) if _PADDLE1 is not None else np.array([0.0,0.0,0.0])
    paddle2_pos = np.array(pyb.getBasePositionAndOrientation(_PADDLE2.body_id)[0]) if _PADDLE2 is not None else np.array([0.0,0.0,0.0])

    collision_events = []
    for idx, p in enumerate(points):
        try:
            pyb.resetBasePositionAndOrientation(body_id, p, pyb.getQuaternionFromEuler([0, 0, 0]))
        except Exception:
            pass
        try:
            pyb.stepSimulation()
        except pyb.error:
            # physics server disconnected; stop animation
            return collision_events

        # Check collisions
        # table: register but do NOT interrupt animation (debounced)
        table_now = check_collision_with_table(p)
        if table_now and not active_collisions["table"]:
            collision_events.append(("table", idx, p))
            active_collisions["table"] = True
            print(f"Collision event: table at index {idx} pos {p}")
        if not table_now and active_collisions["table"]:
            active_collisions["table"] = False

        # net: register but do NOT interrupt animation (debounced)
        net_now = check_collision_with_net(p)
        if net_now and not active_collisions["net"]:
            collision_events.append(("net", idx, p))
            active_collisions["net"] = True
            print(f"Collision event: net at index {idx} pos {p}")
        if not net_now and active_collisions["net"]:
            active_collisions["net"] = False

        # paddle1
        p1_now = check_collision_with_paddle(p, paddle1_pos)
        if p1_now and not active_collisions["paddle1"]:
            collision_events.append(("paddle1", idx, p))
            active_collisions["paddle1"] = True
            print(f"Collision event: paddle1 at index {idx} pos {p}")
            # interrupt so the hit can be handled (ball will be re-shot)
            return collision_events
        if not p1_now and active_collisions["paddle1"]:
            active_collisions["paddle1"] = False

        # paddle2
        p2_now = check_collision_with_paddle(p, paddle2_pos)
        if p2_now and not active_collisions["paddle2"]:
            collision_events.append(("paddle2", idx, p))
            active_collisions["paddle2"] = True
            print(f"Collision event: paddle2 at index {idx} pos {p}")
            return collision_events
        if not p2_now and active_collisions["paddle2"]:
            active_collisions["paddle2"] = False

        time.sleep(step_time)

    # demo() returns after setup; GUI loop is handled by start_env() / __main__
    return None


def start_env(start_gui=False):
    """Set up pybullet, table and paddles. Call once before `rally()`.

    This function mirrors what `demo()` does for initialization but leaves
    the ball uncreated. It returns True on success.
    """
    # Always start in graphical GUI mode but disable HUD elements so the
    # viewer sees the scene without the PyBullet HUD/overlays.
    cid = pyb.connect(pyb.GUI)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(1.0/120.0)

    # Disable GUI elements and shadows/previews so the HUD and overlays
    # are not shown in the viewer.
    try:
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
    except Exception:
        pass
    try:
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_SHADOWS, 0)
    except Exception:
        pass
    # Also disable preview windows if available
    for flag in ("COV_ENABLE_RGB_BUFFER_PREVIEW", "COV_ENABLE_DEPTH_BUFFER_PREVIEW", "COV_ENABLE_SEGMENTATION_MARK_PREVIEW"):
        try:
            val = getattr(pyb, flag)
            try:
                pyb.configureDebugVisualizer(val, 0)
            except Exception:
                pass
        except Exception:
            pass

    sc = ShapeCache()
    tbl = Table(sc)
    tbl.create()

    paddle_height = CFG.TABLE_H + 0.1
    p1 = Paddle(sc, base_pos=[-CFG.TABLE_L/2 + 0.2, 0.0, paddle_height], color=(1, 0.2, 0.2, 1))
    p1.create()
    p2 = Paddle(sc, base_pos=[CFG.TABLE_L/2 - 0.2, 0.0, paddle_height], color=(0.2, 0.2, 1.0, 1))
    p2.create()

    global _CLIENT, _SC, _TABLE, _PADDLE1, _PADDLE2
    _CLIENT = cid
    _SC = sc
    _TABLE = tbl
    _PADDLE1 = p1
    _PADDLE2 = p2

    # start background input polling so keyboard controls work automatically
    try:
        _start_input_thread()
    except Exception:
        pass

    return True


def stop_env():
    global _CLIENT
    # stop input thread if running
    try:
        _stop_input_thread()
    except Exception:
        pass
    try:
        pyb.disconnect()
    except Exception:
        pass
    _CLIENT = None


def _input_thread_loop(stop_event: threading.Event, poll_interval: float = 0.05):
    """Background loop that polls player controllers for keyboard input."""
    while not stop_event.is_set():
        try:
            try:
                import player1 as p1
                p1.update_from_keyboard()
            except Exception:
                pass
            try:
                import player2 as p2
                p2.update_from_keyboard()
            except Exception:
                pass
        except Exception:
            # swallow exceptions to keep thread alive
            pass
        time.sleep(poll_interval)


def _start_input_thread():
    global _INPUT_THREAD, _INPUT_THREAD_STOP
    if _INPUT_THREAD is not None and _INPUT_THREAD.is_alive():
        return
    _INPUT_THREAD_STOP = threading.Event()
    _INPUT_THREAD = threading.Thread(target=_input_thread_loop, args=(_INPUT_THREAD_STOP,), daemon=True)
    _INPUT_THREAD.start()


def _stop_input_thread(timeout: float = 1.0):
    global _INPUT_THREAD, _INPUT_THREAD_STOP
    if _INPUT_THREAD_STOP is None:
        return
    try:
        _INPUT_THREAD_STOP.set()
        if _INPUT_THREAD is not None:
            _INPUT_THREAD.join(timeout)
    finally:
        _INPUT_THREAD = None
        _INPUT_THREAD_STOP = None


if __name__ == '__main__':
    # When run as a script, start the environment in GUI mode and keep the
    # process alive so the user can call `rally()` from the same session or
    # simply observe the scene. Exit with Ctrl+C.
    try:
        start_env(start_gui=True)
        print("Environment started. Call `rally()` to begin a rally, Ctrl+C to exit.")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down environment")
    finally:
        try:
            stop_env()
        except Exception:
            pass
