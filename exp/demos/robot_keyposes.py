"""Add keyposes.json (exp/demos/keyposes.py) to each installed robot expert episode.

Replays each expert's recorded actions in the official environment from its recorded initial state
(as services/dexjoco/demo_tools.py does) to read the task objects' poses at every step, then
summarizes the recording's flange poses, hand joints and those object poses into key moments.
The expert directories are read-only for agents; this rewrites keyposes.json and refreshes the
task manifest's file hashes.

Run it with the host Python on a GPU node; it re-runs itself in the simulator environment:
    $RAX_PYTHON exp/demos/robot_keyposes.py --gpu <GPU uuid> --task hammer_nail [--task ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(Path(__file__).resolve().parent)]

from keyposes import quat_wxyz_to_matrix, summarize  # noqa: E402


def matrix_to_rotvec(m):
    angle = np.arccos(np.clip((np.trace(m) - 1) / 2, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:   # 180 degrees: axis from the symmetric part
        axis = np.sqrt(np.clip((np.diag(m) + 1) / 2, 0, None))
        axis *= np.sign([1, m[0, 1] + 1e-12, m[0, 2] + 1e-12])
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


def object_traces(task, demo):
    """Replay one expert; {label: (N, 7) pose per recorded step} for the task's tracked objects."""
    import zarr
    from dexjoco.tasks import CONFIG_MAPPING
    from dexjoco.tasks.state_restorers import restore_initial_state
    from services.dexjoco import tasks as T
    from services.dexjoco.objects import TAIL, read_live

    info = T.task_info(task)
    root = zarr.open(str(demo / "replay.zarr"), mode="r")
    actions = np.asarray(root["data"]["action"])
    initial = np.asarray(root["data"]["state"][0]).ravel()
    sources = {label: source for label, size, source in TAIL[info["env_name"]] if source and size == 7}
    config = CONFIG_MAPPING[info["env_name"]]()
    env = config.get_environment(policy_mode=True, render_mode="rgb_array", randomize=info["regime"] == "rand_full",
                                 randomize_dynamics=False, seed=0)
    traces = {label: [] for label in sources}
    try:
        env.reset()
        restore_initial_state(env, info["env_name"], config, initial)
        raw = env.unwrapped
        for row in actions:
            row = np.asarray(row, dtype=np.float64)
            # Poses before this step's action, aligned with the recorded state row.
            for label, source in sources.items():
                traces[label].append(read_live(raw._data, source)[:7])
            raw.step({"right": row[:23], "left": row[23:46]} if row.shape == (46,) else row)
    finally:
        env.close()
    return {label: np.asarray(values) for label, values in traces.items()}


def arm_tracks(state, dual):
    """(label, position, quaternion, hand) per arm from the official state rows."""
    if dual:
        return [("right", state[:, 0:3], state[:, 3:7], state[:, 14:30]),
                ("left", state[:, 7:10], state[:, 10:14], state[:, 30:46])]
    return [("right", state[:, 0:3], state[:, 3:7], state[:, 7:23])]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--gpu", help="simulator GPU: re-run this script in the simulator environment on it")
    args = parser.parse_args()
    if args.gpu:
        import subprocess
        from services.dexjoco.process import dexjoco_python, simulator_environment
        command = [dexjoco_python(), __file__, *(f"--task={task}" for task in args.task)]
        raise SystemExit(subprocess.run(command, cwd=PROJECT, env=simulator_environment(args.gpu)).returncode)
    from services.dexjoco import tasks as T
    from services.dexjoco.demonstrations import demos_root

    for task in args.task:
        task_root = demos_root() / task
        manifest = json.loads((task_root / "manifest.json").read_text())
        dual = T.task_info(task)["dual_arm"]
        for name in manifest["experts"]:
            expert = task_root / "experts" / name
            data = np.load(expert / "trajectory.npz")
            state = np.asarray(data["state"], dtype=float).reshape(len(data["state"]), -1)
            objects = object_traces(task, expert)
            arms = {}
            for arm, position, quat, hand in arm_tracks(state, dual):
                rotvec = np.stack([matrix_to_rotvec(quat_wxyz_to_matrix(q)) for q in quat])
                arms[arm] = summarize(hz=T.CONTROL_HZ, position=position, rotvec=rotvec, hand=hand,
                                      objects=objects, source="robot teleoperation (same robot and simulator)")
            out = {"task": task, "episode": name, "arms": arms,
                   "frames": "wrist_* is the flange pose in world metres (the dexjoco_move_ee frame); "
                             "hand_joints_rad is the dexjoco_hand joint order; object poses are this "
                             "recording's layout, not yours"}
            expert.chmod(0o755)
            target = expert / "keyposes.json"
            if target.exists():
                target.chmod(0o644)
            target.write_text(json.dumps(out, indent=1) + "\n")
            target.chmod(0o444)
            expert.chmod(0o555)
            manifest["files"][name]["keyposes.json"] = sha256(target)
            print(json.dumps({"task": task, "episode": name,
                              "moments": {arm: [(m["moment"], m["step"]) for m in s["moments"]]
                                          for arm, s in arms.items()},
                              "contact_cue": {arm: s["contact_cue"] for arm, s in arms.items()}}), flush=True)
        (task_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
