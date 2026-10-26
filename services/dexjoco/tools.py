"""DexJoCo MCP tool schemas, host-side request validation and depth-pixel geometry.

Pure Python + numpy: safe to import in the host harness interpreter. The action and state
layouts are the official evaluation interface (``DexJoCoOpenPIEnv``): actions are the policy's
rotation-vector rows, states the environment's first 23 (single arm) or 46 (dual arm) values.
"""

from __future__ import annotations

import math

import numpy as np

from services.benchmarks import DEXJOCO_TOOLS
from services.dexjoco import rotation_math as RM
from services.simulator.io import RejectedBeforeAction, reject_unless

STATUS, OBSERVE = "dexjoco_status", "dexjoco_observe"
STEP, HOLD = "dexjoco_step", "dexjoco_hold"
# Target-level motion (agent as policy): the server turns a goal into official rows.
MOVE_EE, HAND = "dexjoco_move_ee", "dexjoco_hand"
PIXEL_TO_POSITION = "dexjoco_pixel_to_position"
ROTATION_MATH = RM.TOOL
PREVIEW_DESCRIBE, PREVIEW_RUN = "dexjoco_preview_describe", "dexjoco_preview_run"
PREVIEW_CAMERAS = ("preview_front", "preview_side", "preview_top")
# Robot-only preview: never advances the episode or counts against its step limit.
PREVIEW_TOOLS = frozenset({PREVIEW_DESCRIBE, PREVIEW_RUN})

ROBOT_TOOLS = frozenset({STATUS, OBSERVE, STEP, HOLD, PIXEL_TO_POSITION, ROTATION_MATH, *PREVIEW_TOOLS, MOVE_EE, HAND})
assert ROBOT_TOOLS == DEXJOCO_TOOLS, "services/benchmarks.py and the DexJoCo tool set disagree"
MUTATING = frozenset({STEP, HOLD, MOVE_EE, HAND})
# Target-level motion: flange speed per 20 ms step (m, rad) for each speed profile, the settle
# check, and caps that keep one call from running away.
SPEEDS = {"transit": (0.008, 0.04), "precision": (0.0025, 0.015)}
SETTLE_CHUNK = 10
# The official flange controller converges slowly (time constant about 20 steps). While settling,
# the command leads the target by LEAD_GAIN times the measured error (at most MAX_LEAD_M, so contact
# is not pushed into without bound), which shortens that several-fold.
LEAD_GAIN = 1.5
MAX_LEAD_M = 0.03
# Stalled: the flange moved less than this over a settle chunk without getting closer.
REST_M = 0.002
ROTATION_TOLERANCE_RAD = 0.05
MAX_MOTION_STEPS = 600
# A hand joint whose measured angle stays this far from its target is reported as blocked
# (usually by contact with an object).
BLOCKED_JOINT_RAD = 0.15
MAX_ACTION_ROWS = 150
# Every FRAME_STRIDE-th executed step is saved as a frame (plus the final step).
FRAME_STRIDE = 10
# Every camera name an official evaluation config maps (env image keys).
CAMERAS = ("front", "random_camera", "ego", "ego_right", "wrist", "wrist_left", "wrist_right")
# Geometry-only fields removed by the RGB-only profile.
GEOMETRY_KEYS = frozenset({"camera_poses", "camera_intrinsics", *(f"{camera}_depth_m" for camera in CAMERAS)})

HAND_JOINTS = tuple(f"{finger}{i}" for finger in ("ff", "mf", "rf", "th") for i in range(4))
# The official left Allegro hand reports and takes its joints mirrored: rf, mf, ff, th.
LEFT_HAND_JOINTS = tuple(f"{finger}{i}" for finger in ("rf", "mf", "ff", "th") for i in range(4))


def action_layout(dual_arm):
    """Official policy action rows (DexJoCoOpenPIEnv.step)."""
    if dual_arm:
        return {"dim": 44, "fields": [
            {"slice": [0, 3], "name": "right_flange_position_m", "frame": "world"},
            {"slice": [3, 6], "name": "right_flange_rotvec_rad", "frame": "world", "note": "axis-angle"},
            {"slice": [6, 22], "name": "right_hand_joint_targets_rad", "joints": list(HAND_JOINTS)},
            {"slice": [22, 25], "name": "left_flange_position_m", "frame": "world"},
            {"slice": [25, 28], "name": "left_flange_rotvec_rad", "frame": "world", "note": "axis-angle"},
            {"slice": [28, 44], "name": "left_hand_joint_targets_rad", "joints": list(LEFT_HAND_JOINTS)}],
            "semantics": "absolute targets; the arm tracks the flange pose with the official operational-space "
                         "controller, the hands track joint targets with the official position actuators"}
    return {"dim": 22, "fields": [
        {"slice": [0, 3], "name": "flange_position_m", "frame": "world"},
        {"slice": [3, 6], "name": "flange_rotvec_rad", "frame": "world", "note": "axis-angle"},
        {"slice": [6, 22], "name": "hand_joint_targets_rad", "joints": list(HAND_JOINTS)}],
        "semantics": "absolute targets; the arm tracks the flange pose with the official operational-space "
                     "controller, the hand tracks joint targets with the official position actuators"}


def state_layout(dual_arm):
    """The official evaluation state (env observation `state`, first 23 or 46 values)."""
    if dual_arm:
        return {"dim": 46, "fields": [
            {"slice": [0, 3], "name": "right_flange_position_m", "frame": "world"},
            {"slice": [3, 7], "name": "right_flange_quaternion_wxyz", "frame": "world"},
            {"slice": [7, 10], "name": "left_flange_position_m", "frame": "world"},
            {"slice": [10, 14], "name": "left_flange_quaternion_wxyz", "frame": "world"},
            {"slice": [14, 30], "name": "right_hand_joint_positions_rad", "joints": list(HAND_JOINTS)},
            {"slice": [30, 46], "name": "left_hand_joint_positions_rad", "joints": list(LEFT_HAND_JOINTS)}]}
    return {"dim": 23, "fields": [
        {"slice": [0, 3], "name": "flange_position_m", "frame": "world"},
        {"slice": [3, 7], "name": "flange_quaternion_wxyz", "frame": "world"},
        {"slice": [7, 23], "name": "hand_joint_positions_rad", "joints": list(HAND_JOINTS)}]}


def tool_definitions():
    return [
        {"name": STATUS, "description": (
            "Episode status, step budget, official action and state layouts, cameras and instruction. No physics."),
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": OBSERVE, "description": (
            "Current official DexJoCo observation: the evaluation cameras' RGB (640x640) and depth attachments, "
            "calibration, the official state vector (flange poses and hand joints) and the instruction. "
            "No physics; the observation is cached after the last motion."),
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": STEP, "description": (
            f"Execute 1..{MAX_ACTION_ROWS} rows of the official DexJoCo policy action, one environment step each "
            f"({CONTROL_TEXT}): single arm 22-D [flange xyz (m, world), flange rotation vector (rad, world), "
            "16 Allegro joint targets (rad)]; dual arm 44-D [right xyz, right rotvec, right hand 16, left xyz, "
            "left rotvec, left hand 16]. Targets are absolute; status gives the exact layout. Stops early when the "
            f"episode ends. Returns the final frame and saves every {FRAME_STRIDE}th frame."),
         "inputSchema": {"type": "object", "required": ["actions"], "additionalProperties": False, "properties": {
             "actions": {"type": "array", "minItems": 1, "maxItems": MAX_ACTION_ROWS,
                         "items": {"type": "array", "items": {"type": "number"}, "minItems": 22, "maxItems": 44}}}}},
        {"name": HOLD, "description": (
            f"Hold the current pose for 1..{MAX_ACTION_ROWS} environment steps with the official stay(): the first "
            "held step targets the current measured flange pose and hand joints, consecutive holds keep that target. "
            "Each held step reports the row it sent as its executed_action and counts against the episode limit."),
         "inputSchema": {"type": "object", "required": ["steps"], "additionalProperties": False, "properties": {
             "steps": {"type": "integer", "minimum": 1, "maximum": MAX_ACTION_ROWS}}}},
        {"name": MOVE_EE, "description": (
            "Move one arm's flange (end effector) to a target pose; the server interpolates the motion into "
            f"official action rows ({CONTROL_TEXT}), runs them with the official flange controller, then holds the "
            "target until the measured flange is within tolerance, max_settle_steps run out, or it stalls short "
            "of the target (stalled: contact or out of reach). Give position_m "
            "(world) or delta_m (world offset from the measured flange position), and/or rotation (world; default: "
            "keep the measured orientation). hand_joints_rad (optional) moves that arm's hand to new joint targets "
            "along the way; otherwise the hand keeps its last commanded targets, so a grasp stays closed. speed "
            "transit is fast (about 0.24 m/s), precision slow (about 0.075 m/s) for approaches and contact. The "
            "other arm (dual-arm tasks) keeps its last commanded targets. Replies with the reached flange pose and "
            "its errors, the steps used, and the final frame; every step counts against the episode limit."),
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "arm": _ARM,
             "position_m": {**_vector(3), "description": "flange target position, world frame (m)"},
             "delta_m": {**_vector(3), "description": "world-frame offset from the measured flange position (m)"},
             "rotation": _ROTATION,
             "hand_joints_rad": {**_vector(16), "description": "optional hand joint targets (rad), that arm's order"},
             "speed": {"type": "string", "enum": list(SPEEDS), "default": "transit"},
             "position_tolerance_m": {"type": "number", "minimum": 0.002, "maximum": 0.05, "default": 0.01},
             "max_settle_steps": {"type": "integer", "minimum": 0, "maximum": 120, "default": 30}}}},
        {"name": HAND, "description": (
            "Move one arm's 16 Allegro hand joints to absolute targets (rad, that hand's joint order; status gives "
            "it) over `steps` environment steps, with the arm holding its last commanded flange target, then hold "
            "settle_steps more. Replies with the measured joints, each joint's tracking error and the joints still "
            f"more than {BLOCKED_JOINT_RAD} rad from their targets (blocked, usually by contact), and the final "
            "frame. The targets stay commanded afterwards, so a closed hand keeps squeezing through later moves."),
         "inputSchema": {"type": "object", "required": ["joints_rad"], "additionalProperties": False, "properties": {
             "arm": _ARM,
             "joints_rad": {**_vector(16), "description": "absolute hand joint targets (rad)"},
             "steps": {"type": "integer", "minimum": 1, "maximum": 60, "default": 15},
             "settle_steps": {"type": "integer", "minimum": 0, "maximum": 30, "default": 5}}}},
        {"name": PIXEL_TO_POSITION, "description": (
            "Back-project 1..32 pixels of the current observation's depth images into world-frame points "
            "(metres), using each camera's calibration. Pass the observation's step_id."),
         "inputSchema": {"type": "object", "required": ["observation_step_id", "selections"],
                         "additionalProperties": False, "properties": {
                             "observation_step_id": {"type": "integer"},
                             "selections": {"type": "array", "minItems": 1, "maxItems": 32, "items": {
                                 "type": "object", "required": ["camera", "u", "v"], "additionalProperties": False,
                                 "properties": {"camera": {"type": "string", "enum": list(CAMERAS)},
                                                "u": {"type": "number"}, "v": {"type": "number"}}}},
                             "depth_window_radius": {"type": "integer", "minimum": 0, "maximum": 5}}}},
        {"name": ROTATION_MATH, "description": (
            "Rotation and pose arithmetic for dexjoco_step rows, converted exactly as the official row conversion "
            "does (scipy Rotation: rotation vector <-> quaternion wxyz). Operations: convert_rotation (to "
            "quaternion_wxyz, rotation_vector and matrix); compose_pose (pose_a, then pose_b expressed in pose_a's "
            "frame); relative_pose (pose_b expressed in pose_a's frame); flange_target (the flange pose that puts a "
            "point fixed to the hand, given by its offset in the flange frame, at point_target_m with the flange "
            "orientation `rotation`); row_from_state (the hold row the official stay() builds from a state "
            "vector). Poses reply with row_prefix: position then rotation vector, the flange part of a row. "
            "near_rotation_vector selects the equivalent rotation vector closest to it, so consecutive rows do "
            "not jump across the angle-pi boundary. Pure geometry: needs no episode and reads no scene state."),
         "inputSchema": {"type": "object", "required": ["operation"], "additionalProperties": False, "properties": {
             "operation": {"type": "string", "enum": list(RM.OPERATIONS)},
             "rotation": _ROTATION, "pose_a": _POSE, "pose_b": _POSE,
             "point_target_m": {**_vector(3), "description": "flange_target: world position the hand point must reach"},
             "point_offset_m": {**_vector(3), "description": "flange_target: the hand point in the flange frame (m)"},
             "near_rotation_vector": {**_vector(3), "description": "e.g. the previous row's rotation vector"},
             "state": {"type": "array", "items": _NUMBER, "minItems": 23, "maxItems": 46,
                       "description": "row_from_state: the official state vector (23 or 46 values)"}}}},
        {"name": PREVIEW_DESCRIBE, "description": (
            "Contract of the robot-only motion preview: robot model, control, cameras, inputs, outputs and limits. "
            "Starts nothing."),
         "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": PREVIEW_RUN, "description": (
            "Dry-run dexjoco_step rows on a robot-only simulator: this task's Panda + Allegro model under the "
            f"official control (the same flange controller and gains, {CONTROL_TEXT}, the same hand servos) on a "
            "floor plane, watched by fixed third-person cameras. No table, objects or contact with the scene. The "
            "start defaults to the episode's measured state vector; initial_state (an official state vector) "
            "overrides it. Each run first drives the arm from its home posture to the start flange pose (not shown; "
            "reported as start_settle). Returns the start and end state vectors, saves every "
            f"{FRAME_STRIDE}th row as a frame and attaches the final frames. Never moves the episode or spends its "
            "steps."),
         "inputSchema": {"type": "object", "required": ["actions"], "additionalProperties": False, "properties": {
             "actions": {"type": "array", "minItems": 1, "maxItems": MAX_ACTION_ROWS,
                         "items": {"type": "array", "items": _NUMBER, "minItems": 22, "maxItems": 44}},
             "initial_state": {"type": "array", "items": _NUMBER, "minItems": 23, "maxItems": 46,
                               "description": "Official state vector (23 single arm, 46 dual arm); default: the "
                                              "episode's measured state."}}}},
    ]


CONTROL_TEXT = "30 Hz, 20 ms of physics per step"
_NUMBER = {"type": "number"}


def _vector(n):
    return {"type": "array", "items": _NUMBER, "minItems": n, "maxItems": n}


_ROTATION = {
    "type": "object", "minProperties": 1, "maxProperties": 1, "additionalProperties": False,
    "description": "Exactly one representation (world frame unless a pose says otherwise).",
    "properties": {
        "quaternion_wxyz": _vector(4),
        "rotation_vector": {**_vector(3), "description": "axis times angle (rad), as in action rows"},
        "matrix": {"type": "array", "items": _vector(3), "minItems": 3, "maxItems": 3, "description": "rows of R"},
        "axis_angle": {"type": "object", "required": ["axis", "angle_rad"], "additionalProperties": False,
                       "properties": {"axis": _vector(3), "angle_rad": _NUMBER}}}}
_ARM = {"type": "string", "enum": ["right", "left"], "default": "right",
        "description": "dual-arm tasks: which arm; single-arm tasks have only the right arm"}
_POSE = {"type": "object", "minProperties": 1, "additionalProperties": False,
         "description": "position_m (m) and rotation; a missing position is zero, a missing rotation the identity.",
         "properties": {"position_m": _vector(3), "rotation": _ROTATION}}


def _only(arguments, allowed):
    unknown = set(arguments) - set(allowed)
    reject_unless(not unknown, f"Unknown arguments: {sorted(unknown)}")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_actions(actions, dim=None):
    reject_unless(isinstance(actions, list) and 1 <= len(actions) <= MAX_ACTION_ROWS,
                  f"actions must be a list of 1..{MAX_ACTION_ROWS} rows")
    width = len(actions[0]) if isinstance(actions[0], list) else None
    reject_unless(width in (22, 44), "each action row must have 22 (single arm) or 44 (dual arm) numbers")
    if dim is not None:
        reject_unless(width == dim, f"this task takes {dim}-D action rows, got {width}")
    for row in actions:
        reject_unless(isinstance(row, list) and len(row) == width, "all action rows must have the same length")
        reject_unless(all(_finite(v) for v in row), "action rows must contain finite numbers")


def validate_state(state, dim):
    """An official state vector of this robot: finite, with non-zero flange quaternions."""
    reject_unless(isinstance(state, list) and len(state) == dim and all(_finite(v) for v in state),
                  f"initial_state must be the official {dim}-D state vector of this robot (dexjoco_status state_layout)")
    for start in ((3,) if dim == 23 else (3, 10)):
        reject_unless(math.hypot(*state[start:start + 4]) > 1e-6,
                      f"initial_state[{start}:{start + 4}] must be a non-zero flange quaternion [w, x, y, z]")


def _arm(arguments, action_dim):
    arm = arguments.get("arm", "right")
    reject_unless(arm in ("right", "left"), "arm must be 'right' or 'left'")
    reject_unless(arm == "right" or action_dim != 22, "this task has one arm: arm must be 'right'")


def arm_slices(dual_arm, arm):
    """(row flange, row hand, state position, state quaternion, state hand) slices of one arm."""
    if not dual_arm:
        return slice(0, 6), slice(6, 22), slice(0, 3), slice(3, 7), slice(7, 23)
    if arm == "right":
        return slice(0, 6), slice(6, 22), slice(0, 3), slice(3, 7), slice(14, 30)
    return slice(22, 28), slice(28, 44), slice(7, 10), slice(10, 14), slice(30, 46)


def validate_request(name, arguments, *, action_dim=None):
    """Complete schema/shape validation before dispatch; raises RejectedBeforeAction."""
    reject_unless(isinstance(arguments, dict), "arguments must be an object")
    if name in (STATUS, OBSERVE):
        _only(arguments, ())
    elif name == STEP:
        _only(arguments, ("actions",))
        validate_actions(arguments.get("actions"), action_dim)
    elif name == HOLD:
        _only(arguments, ("steps",))
        steps = arguments.get("steps")
        reject_unless(type(steps) is int and 1 <= steps <= MAX_ACTION_ROWS, f"steps must be an integer 1..{MAX_ACTION_ROWS}")
    elif name == MOVE_EE:
        _only(arguments, ("arm", "position_m", "delta_m", "rotation", "hand_joints_rad", "speed",
                          "position_tolerance_m", "max_settle_steps"))
        _arm(arguments, action_dim)
        reject_unless(not ("position_m" in arguments and "delta_m" in arguments), "give position_m or delta_m, not both")
        reject_unless(any(k in arguments for k in ("position_m", "delta_m", "rotation", "hand_joints_rad")),
                      "give a target: position_m, delta_m, rotation and/or hand_joints_rad")
        for key, size in (("position_m", 3), ("delta_m", 3), ("hand_joints_rad", 16)):
            if key in arguments:
                RM._vector(arguments[key], size, key)
        if "rotation" in arguments:
            RM._rotation(arguments["rotation"], "rotation")
        reject_unless(arguments.get("speed", "transit") in SPEEDS, f"speed must be one of {list(SPEEDS)}")
        tolerance = arguments.get("position_tolerance_m", 0.01)
        reject_unless(_finite(tolerance) and 0.002 <= tolerance <= 0.05, "position_tolerance_m must be 0.002..0.05")
        settle = arguments.get("max_settle_steps", 30)
        reject_unless(type(settle) is int and 0 <= settle <= 120, "max_settle_steps must be an integer 0..120")
    elif name == HAND:
        _only(arguments, ("arm", "joints_rad", "steps", "settle_steps"))
        _arm(arguments, action_dim)
        RM._vector(arguments.get("joints_rad"), 16, "joints_rad")
        steps, settle = arguments.get("steps", 15), arguments.get("settle_steps", 5)
        reject_unless(type(steps) is int and 1 <= steps <= 60, "steps must be an integer 1..60")
        reject_unless(type(settle) is int and 0 <= settle <= 30, "settle_steps must be an integer 0..30")
    elif name == ROTATION_MATH:
        RM.validate(arguments)
    elif name == PREVIEW_DESCRIBE:
        _only(arguments, ())
    elif name == PREVIEW_RUN:
        _only(arguments, ("actions", "initial_state"))
        validate_actions(arguments.get("actions"), action_dim)
        if "initial_state" in arguments:
            validate_state(arguments["initial_state"], 46 if len(arguments["actions"][0]) == 44 else 23)
    elif name == PIXEL_TO_POSITION:
        _only(arguments, ("observation_step_id", "selections", "depth_window_radius"))
        reject_unless(type(arguments.get("observation_step_id")) is int, "observation_step_id must be an integer")
        selections = arguments.get("selections")
        reject_unless(isinstance(selections, list) and 1 <= len(selections) <= 32, "selections must have 1..32 items")
        for item in selections:
            reject_unless(isinstance(item, dict) and set(item) == {"camera", "u", "v"}, "selection needs camera, u, v")
            reject_unless(item["camera"] in CAMERAS, f"unknown camera {item['camera']!r}")
            reject_unless(_finite(item["u"]) and _finite(item["v"]), "pixel coordinates must be finite numbers")
        radius = arguments.get("depth_window_radius", 1)
        reject_unless(type(radius) is int and 0 <= radius <= 5, "depth_window_radius must be 0..5")
    else:
        raise RejectedBeforeAction(f"Unknown DexJoCo tool: {name}")


def quaternion_matrix(q_wxyz):
    w, x, y, z = (float(v) for v in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pixel_to_world(depth_m, intrinsics, pose, u, v, radius=1):
    """Back-project one pixel of a linear-depth image into the world frame.

    MuJoCo cameras look down their -Z axis with +Y up (OpenGL); images are row 0 at the top.
    Depth is the official linearised z-buffer: distance along the optical axis. Uses the
    median valid depth in a (2r+1)^2 window.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    height, width = depth.shape[:2]
    reject_unless(0 <= u < width and 0 <= v < height, f"pixel ({u}, {v}) outside {width}x{height}")
    iu, iv = int(round(u)), int(round(v))
    window = depth[max(0, iv - radius):iv + radius + 1, max(0, iu - radius):iu + radius + 1]
    valid = window[np.isfinite(window) & (window > 0)]
    if valid.size == 0:
        return {"valid": False, "reason": "no valid depth near the pixel"}
    z = float(np.median(valid))
    K = np.asarray(intrinsics["matrix"], dtype=np.float64)
    point_camera = np.array([(u - K[0, 2]) / K[0, 0] * z, -(v - K[1, 2]) / K[1, 1] * z, -z])
    point = quaternion_matrix(pose["quaternion_wxyz"]) @ point_camera + np.asarray(pose["position_m"], dtype=np.float64)
    return {"valid": True, "position_m": point.tolist(), "depth_m": z, "frame": "world",
            "depth_spread_m": float(valid.max() - valid.min())}


def world_to_pixel(point, intrinsics, pose):
    """Project a world-frame point into a camera (inverse of pixel_to_world)."""
    local = quaternion_matrix(pose["quaternion_wxyz"]).T @ (np.asarray(point, dtype=np.float64)
                                                            - np.asarray(pose["position_m"], dtype=np.float64))
    if local[2] >= 0:
        return None
    z = -local[2]
    K = np.asarray(intrinsics["matrix"], dtype=np.float64)
    return float(K[0, 0] * local[0] / z + K[0, 2]), float(-K[1, 1] * local[1] / z + K[1, 2]), float(z)
