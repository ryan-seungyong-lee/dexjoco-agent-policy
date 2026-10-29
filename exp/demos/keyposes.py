"""Key moments of one demonstration, in the form an agent can turn into motion targets (numpy only).

A demonstration (robot teleoperation or a tracked human video) becomes a short list of moments:
start, (preshape), approach, pre-contact, contact, (grasp), (lift), (release), end. Each moment gives

- the wrist/flange pose, and its offset from the pose at contact (both sources have it);
- the hand posture (robot: the Allegro joints it held; human: the Allegro retarget) and its mean
  joint flexion (0 = flat);
- per tracked object (robot demos only; human videos have no object poses): the flange position
  relative to the object, in world axes and in the object's own heading (yaw) frame, and how far
  the object has moved so far.

``summarize`` takes plain arrays so the robot exporter (robot_keyposes.py, run in the simulator
environment) and a human-demo builder can share it.
"""
from __future__ import annotations

import numpy as np

APPROACH_M = 0.10       # approach: last time the wrist was this far from where it makes contact
PRE_CONTACT_M = 0.03    # pre-contact: ... this far
MOVED_M = 0.01          # an object counts as moved once it is this far from its start
LIFT_M = 0.02           # a lift moment is reported when an object rises at least this much
NEAR_M = 0.25           # an object's motion counts as contact only with the wrist this close to it
SETTLE_STEPS = 10       # objects settle after reset for a few steps
PRESHAPE_RAD = 0.15     # mean joint flexion gained before contact that counts as a pre-shape
GRASP_SECONDS = 3.0     # time after contact searched for the grasp
GRASP_RAD = 0.03        # extra mean flexion after contact that counts as closing on the object
RELEASE_RAD = 0.10      # opening below the contact posture by this much counts as a release


def quat_wxyz_to_matrix(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec_to_matrix(r):
    r = np.asarray(r, dtype=float)
    angle = np.linalg.norm(r)
    if angle < 1e-12:
        return np.eye(3)
    k = r / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def yaw_of(matrix):
    return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def _r(values, digits=3):
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def _last_before(distance, end, threshold):
    """Last index before ``end`` where ``distance`` was at least ``threshold`` (0 if never)."""
    far = np.nonzero(distance[:end] >= threshold)[0]
    return int(far[-1]) if len(far) else 0


def summarize(*, hz, position, rotvec, hand, objects=None, contact=None, source):
    """position (N, 3) and rotvec (N, 3): wrist/flange pose per step; hand (N, 16) joint angles.
    objects: {label: (N, 7) position + wxyz quaternion per step} (robot demos), or None.
    contact: (N,) bool when the source measures contact directly (human: contact probability)."""
    position, rotvec, hand = (np.asarray(a, dtype=float) for a in (position, rotvec, hand))
    n = len(position)
    flex = hand.mean(axis=1)
    objects = {label: np.asarray(pose, dtype=float) for label, pose in (objects or {}).items()}
    # Objects settle for a few steps after reset: measure their motion from the settled pose.
    settled = min(SETTLE_STEPS, n - 1)
    moved = {label: np.linalg.norm(pose[:, :3] - pose[settled, :3], axis=1) for label, pose in objects.items()}
    near = {label: np.linalg.norm(position - pose[:, :3], axis=1) <= NEAR_M for label, pose in objects.items()}

    # Contact: the first of an object starting to move while the wrist is near it, and a measured
    # contact; without either, the wrist's closest approach to an object (or its lowest point).
    candidates = {}
    for label, d in moved.items():
        touched = (d >= MOVED_M) & near[label]
        touched[:settled] = False
        if touched.any():
            candidates[f"{label} starts moving"] = int(np.argmax(touched))
    if contact is not None and np.asarray(contact).any():
        candidates["contact measured"] = int(np.argmax(np.asarray(contact)))
    if candidates:
        cue = min(candidates, key=candidates.get)
        k_contact = candidates[cue]
    elif objects:
        label, distance = min(((label, np.linalg.norm(position - pose[:, :3], axis=1))
                               for label, pose in objects.items()), key=lambda item: item[1].min())
        cue, k_contact = f"closest to {label}", int(np.argmin(distance))
    else:
        cue, k_contact = "lowest wrist point", int(np.argmin(position[:, 2]))
    to_contact = np.linalg.norm(position - position[k_contact], axis=1)

    steps = {"start": 0,
             "approach": _last_before(to_contact, k_contact, APPROACH_M),
             "pre_contact": _last_before(to_contact, k_contact, PRE_CONTACT_M),
             "contact": k_contact}
    # The hand usually pre-shapes on the way (flat -> half curled), then closes a little more on
    # the object: preshape = half way to the contact posture, grasp = the most curled soon after.
    if flex[k_contact] - flex[0] >= PRESHAPE_RAD:
        steps["preshape"] = int(np.argmax(flex >= flex[0] + 0.5 * (flex[k_contact] - flex[0])))
    window = flex[k_contact:k_contact + max(1, round(GRASP_SECONDS * hz))]
    k_grasp = k_contact + int(np.argmax(window))
    if window.max() - flex[k_contact] >= GRASP_RAD:
        steps["grasp"] = k_grasp
    opened = np.nonzero(flex[k_grasp:] <= flex[k_contact] - RELEASE_RAD)[0]
    if len(opened):
        steps["release"] = k_grasp + int(opened[0])
    rises = {label: pose[:, 2] - pose[settled, 2] for label, pose in objects.items()}
    lifted = [label for label, rise in rises.items() if rise[k_contact:].max() >= LIFT_M]
    if lifted:
        label = max(lifted, key=lambda name: rises[name][k_contact:].max())
        steps["lift"] = k_contact + int(np.argmax(rises[label][k_contact:]))
    steps["end"] = n - 1
    order = sorted(steps.items(), key=lambda item: (item[1], list(steps).index(item[0])))

    contact_rot = rotvec_to_matrix(rotvec[k_contact])
    moments = []
    for name, k in order:
        rot = rotvec_to_matrix(rotvec[k])
        moment = {
            "moment": name, "step": k, "time_s": round(k / hz, 2),
            "wrist_position_m": _r(position[k]), "wrist_rotvec_rad": _r(rotvec[k]),
            "wrist_from_contact_m": _r(position[k] - position[k_contact]),
            "wrist_from_contact_in_contact_frame_m": _r(contact_rot.T @ (position[k] - position[k_contact])),
            "wrist_yaw_deg": round(np.degrees(yaw_of(rot)), 1),
            "hand_joints_rad": _r(hand[k]), "hand_flexion_mean_rad": round(float(flex[k]), 3),
        }
        if objects:
            rel = {}
            for label, pose in objects.items():
                obj_rot = quat_wxyz_to_matrix(pose[k, 3:7])
                obj_yaw = yaw_of(obj_rot)
                c, s = np.cos(obj_yaw), np.sin(obj_yaw)
                offset = position[k] - pose[k, :3]
                rel[label] = {
                    "object_position_m": _r(pose[k, :3]),
                    "object_yaw_deg": round(np.degrees(obj_yaw), 1),
                    "wrist_minus_object_world_m": _r(offset),
                    "wrist_minus_object_in_object_yaw_frame_m": _r([c * offset[0] + s * offset[1],
                                                                    -s * offset[0] + c * offset[1], offset[2]]),
                    "wrist_yaw_minus_object_yaw_deg": round(float(np.degrees(
                        (yaw_of(rot) - obj_yaw + np.pi) % (2 * np.pi) - np.pi)), 1),
                    "object_moved_m": round(float(moved[label][k]), 3),
                }
            moment["objects"] = rel
        moments.append(moment)

    direction = position[k_contact] - position[steps["approach"]]
    norm = np.linalg.norm(direction)
    speed = np.linalg.norm(np.diff(position, axis=0), axis=1) * hz
    window = slice(steps["pre_contact"], max(k_contact, steps["pre_contact"] + 1))
    return {
        "source": source, "steps": n, "hz": hz, "contact_cue": cue,
        "approach_direction_world": _r(direction / norm) if norm > 1e-6 else None,
        "approach_distance_m": round(float(norm), 3),
        "speed_near_contact_m_per_s": round(float(speed[window].mean()), 3) if len(speed[window]) else None,
        "peak_speed_m_per_s": round(float(speed.max()), 3) if len(speed) else None,
        "moments": moments,
    }
