#!/usr/bin/env python3
"""Build the human in-context demonstrations (expert content human-video / human-full) for a
DexJoCo task from chosen EgoDex clips.

For each clip it writes runtime/reference-demos/dexjoco-human/<task>/experts/<name>/:
  video.mp4            the matching span, 640 px wide, 15 fps
  frames/              8 keyframes + index.json (time, phase label)
  episode.json         source, instruction, hands, span, frame conventions
  hand_trajectory.npz  10 Hz wrist pose, 21 keypoints, finger flexion, aperture, contact
  phases.json          per-hand idle/reach/grasp/manipulate/release/retract segments
  allegro_retarget.npz Allegro joint targets per hand from DexJoCo's own teleoperation
                       retargeting (GeoRT checkpoints of the pinned DexJoCo checkout)
plus manifest.json for the task, and a verify/ directory outside it (keypoint overlay video,
retargeting fit) for a person to check.

Selection file (json): {"task": "fold_glasses", "clips": [{"h5": ".../0.hdf5", "start_s": 0.0,
"end_s": 8.5, "hands": "right"}, ...3 clips]}

Usage: python exp/human_demos/build_human_demos.py --selection sel.json --out <dir> --geort <GeoRT dir>
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = 'dexjoco_human_demos_v1'
import os
# EGODEX_ROOT: the EgoDex train set (partN/<task>/<i>.{mp4,hdf5}); RLHND_ROOT: ':'-separated roots of the
# per-clip hand reconstructions (contact probability, force) that match it.
RLHND_ROOTS = tuple(Path(p) for p in os.environ.get('RLHND_ROOT', '').split(':') if p)
EGODEX_ROOT = Path(os.environ.get('EGODEX_ROOT', 'data/egodex-train'))
TRAJ_HZ, VIDEO_FPS, VIDEO_WIDTH, KEYFRAMES = 10, 15, 640, 8
FINGERS = ('Thumb', 'IndexFinger', 'MiddleFinger', 'RingFinger', 'LittleFinger')
# EgoDex joints in the 21-point order GeoRT/Rokoko use: wrist, then 4 per finger base->tip.
SEGMENTS = {'Thumb': ('Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip')}
SIDES = ('left', 'right')  # hand slot 0 = left, 1 = right (as in RLHND)


def joint_names(side):
    names = [f'{side}Hand']
    for finger in FINGERS:
        parts = SEGMENTS.get(finger, ('Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip'))
        names += [f'{side}{finger}{part}' for part in parts]
    return names


# ------------------------------------------------------------------ geometry
def gravity_frame(camera0):
    """World(ARKit, y up) -> frame anchored at the first head pose: x right, y forward, z up."""
    up = np.array([0.0, 1.0, 0.0])
    forward = camera0[:3, 2].copy()  # EgoDex camera axes are OpenCV-like: +z looks forward
    forward -= forward.dot(up) * up
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    rotation = np.stack([right, forward, up])  # rows: new axes in world coordinates
    return rotation, camera0[:3, 3].copy()


def to_frame(points, rotation, origin):
    return (points - origin) @ rotation.T


def flexion(keypoints):
    """Total curl per finger (radians): sum of angles between consecutive bones."""
    out = np.zeros(keypoints.shape[:-2] + (5,))
    for f in range(5):
        chain = [0] + [1 + 4 * f + i for i in range(4)]
        bones = np.diff(keypoints[..., chain, :], axis=-2)
        bones /= np.linalg.norm(bones, axis=-1, keepdims=True) + 1e-9
        cos = np.clip(np.sum(bones[..., :-1, :] * bones[..., 1:, :], axis=-1), -1, 1)
        out[..., f] = np.arccos(cos).sum(-1)
    return out


def hand_to_canonical(points, is_left):
    """DexJoCo teleoperation's canonical hand frame (teleoperation/rokoko/common.py)."""
    z = points[9] - points[0]
    z /= np.linalg.norm(z) + 1e-9
    y_aux = points[13] - points[5] if is_left else points[5] - points[13]
    y_aux /= np.linalg.norm(y_aux) + 1e-9
    x = np.cross(y_aux, z)
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    basis = np.stack([x, y, z], axis=1)
    return (points - points[0]) @ basis


# ------------------------------------------------------------------ retargeting
class GeoRT:
    """DexJoCo's GeoRT IK (CPU), loaded from the checkpoint directory of one hand."""

    def __init__(self, geort_dir, side):
        sys.path.insert(0, str(geort_dir))
        import torch
        from geort.model import FKModel, IKModel
        from geort.utils.config_utils import load_json, parse_config_joint_limit, parse_config_keypoint_info
        tag = f'dexjoco_{side}_default'
        checkpoint = next(p for p in (Path(geort_dir) / 'checkpoint').iterdir() if p.name.endswith(tag))
        config = load_json(checkpoint / 'config.json')
        info = parse_config_keypoint_info(config)
        self.lower, self.upper = parse_config_joint_limit(config)
        self.human_ids = info['human_id']
        self.torch = torch
        self.ik = IKModel(keypoint_joints=info['joint'])
        self.ik.load_state_dict(torch.load(checkpoint / 'last.pth', map_location='cpu'))
        self.ik.eval()
        self.fk = FKModel(keypoint_joints=info['joint'])
        self.fk.load_state_dict(torch.load(Path(geort_dir) / 'checkpoint' / f'fk_model_allegro_{side}.pth',
                                           map_location='cpu'))
        self.fk.eval()

    def __call__(self, canonical):
        """canonical: (N, 21, 3) -> joints (N, 16) in the dexjoco_step order of that hand, fit error (N,)."""
        torch = self.torch
        with torch.no_grad():
            tips = torch.from_numpy(canonical[:, self.human_ids].astype(np.float32))
            normalized = self.ik(tips)
            joints = (normalized.numpy() / 2 + 0.5) * (self.upper - self.lower) + self.lower
            reached = self.fk(normalized).numpy()
        error = np.linalg.norm(reached - canonical[:, self.human_ids], axis=-1).mean(-1)
        return joints, error


# ------------------------------------------------------------------ contact and phases
def rlhnd_contact(h5_path, frames):
    """Per-hand contact probability (T, 2) and total force from RLHND, if that clip was tracked."""
    relative = Path(h5_path).relative_to(EGODEX_ROOT).with_suffix('.npz')
    for root in RLHND_ROOTS:
        path = root / relative
        if path.is_file():
            data = np.load(path)
            prob = np.asarray(data['contact_prob'], dtype=np.float32)  # (T, 2, 778)
            exist = np.asarray(data['exist'], dtype=np.float32) > 0.5
            # A hand touches when a patch of its surface does: the 95th percentile over vertices.
            contact = np.percentile(prob, 95, axis=-1) * exist
            force = np.asarray(data['force_total'], dtype=np.float32) * exist
            n = min(len(contact), frames)
            pad = frames - n
            return (np.pad(contact[:n], ((0, pad), (0, 0))), np.pad(force[:n], ((0, pad), (0, 0))), str(path))
    return None, None, None


def phases(times, speed, contact, present, threshold=0.5, min_seconds=0.3, max_gap_seconds=1.0):
    """Segments per hand from contact (if any) and wrist speed."""
    result = {}
    dt = times[1] - times[0] if len(times) > 1 else 0.1
    for slot, side in enumerate(SIDES):
        touching = contact[:, slot] > threshold if contact is not None else np.zeros(len(times), bool)
        # Remove contact blips shorter than min_seconds.
        min_len = max(1, int(round(min_seconds / dt)))
        label = np.empty(len(times), dtype=object)
        runs = np.flatnonzero(np.diff(np.r_[0, touching.astype(int), 0]))
        for start, stop in zip(runs[::2], runs[1::2]):
            if stop - start < min_len:
                touching[start:stop] = False
        # Bridge short tracking dropouts (a hand hidden behind the object for a moment).
        seen = present[:, slot].copy()
        gaps = np.flatnonzero(np.diff(np.r_[0, (~seen).astype(int), 0]))
        for start, stop in zip(gaps[::2], gaps[1::2]):
            if 0 < start and stop < len(seen) and stop - start <= int(round(max_gap_seconds / dt)):
                seen[start:stop] = True
        first = np.argmax(touching) if touching.any() else None
        last = len(touching) - 1 - np.argmax(touching[::-1]) if touching.any() else None
        for i in range(len(times)):
            if not seen[i]:
                label[i] = 'absent'
            elif first is None:
                label[i] = 'reach' if speed[i, slot] > 0.05 else 'idle'
            elif i < first:
                label[i] = 'reach' if speed[i, slot] > 0.03 or i >= first - min_len else 'idle'
            elif touching[i]:
                label[i] = 'grasp' if i < first + min_len else 'manipulate'
            elif i > last:
                label[i] = 'release' if i <= last + min_len else ('retract' if speed[i, slot] > 0.03 else 'idle')
            else:
                label[i] = 'manipulate'  # brief gaps between contacts
        segments, start = [], 0
        for i in range(1, len(times) + 1):
            if i == len(times) or label[i] != label[start]:
                segments.append({'phase': label[start], 'start_s': round(float(times[start]), 2),
                                 'end_s': round(float(times[i - 1] + dt), 2)})
                start = i
        result[side] = segments
    return result


# ------------------------------------------------------------------ video
def write_video(video, start, end, out, keyframe_dir, labels):
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    first, last = int(start * fps), int(end * fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    step = max(1, int(round(fps / VIDEO_FPS)))
    writer, frames = None, []
    for index in range(first, last):
        ok, frame = capture.read()
        if not ok:
            break
        if (index - first) % step:
            continue
        frame = cv2.resize(frame, (VIDEO_WIDTH, int(frame.shape[0] * VIDEO_WIDTH / frame.shape[1])),
                           interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        frames.append(((index - first) / fps, frame))
    capture.release()
    writer.release()
    keyframe_dir.mkdir()
    index_entries = []
    for order, k in enumerate(np.linspace(0, len(frames) - 1, KEYFRAMES).astype(int)):
        t, frame = frames[k]
        name = f'frame_{order:02d}.jpg'
        cv2.imwrite(str(keyframe_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        index_entries.append({'file': f'frames/{name}', 'time_s': round(t, 2), 'hands': labels(t)})
    (keyframe_dir / 'index.json').write_text(json.dumps({'frames': index_entries, 'camera': 'head'}, indent=2))
    return len(frames)


def overlay_video(video, start, end, out, keypoints_cam, intrinsic, fps_traj):
    """Keypoints (camera frame, OpenCV) projected on the video, for a person to check tracking."""
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    writer = None
    for index in range(int(start * fps), int(end * fps), 2):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            break
        k = min(int(round((index / fps - start) * fps_traj)), len(keypoints_cam) - 1)
        for slot, color in ((0, (255, 128, 0)), (1, (0, 128, 255))):
            points = keypoints_cam[k, slot]
            if not np.isfinite(points).all() or (points[:, 2] <= 0).any():
                continue
            uv = (points[:, :2] / points[:, 2:3]) * [intrinsic[0, 0], intrinsic[1, 1]] + intrinsic[:2, 2]
            for u, v in uv.astype(int):
                cv2.circle(frame, (u, v), 6, color, -1)
        frame = cv2.resize(frame, (960, 540))
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*'mp4v'), fps / 2, (960, 540))
        writer.write(frame)
    capture.release()
    if writer:
        writer.release()


# ------------------------------------------------------------------ one clip
def build_clip(clip, task, out_dir, verify_dir, geort, mirror):
    h5_path = Path(clip['h5'].replace('EGODEX', str(EGODEX_ROOT), 1))
    with h5py.File(h5_path, 'r') as f:
        attrs = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in f.attrs.items()}
        camera = f['transforms']['camera'][()]
        intrinsic = f['camera']['intrinsic'][()]
        hands = np.stack([np.stack([f['transforms'][name][()] for name in joint_names(side)], axis=1)
                          for side in SIDES], axis=1)  # (T, 2, 21, 4, 4)
        if 'confidences' in f:
            confidence = np.stack([np.stack([f['confidences'][name][()] for name in joint_names(side)], axis=1)
                                   for side in SIDES], axis=1)  # (T, 2, 21)
        else:  # some clips carry no confidences
            confidence = np.ones(hands.shape[:3])
    total = len(camera)
    start_f = int(clip.get('start_s', 0) * 30)
    end_f = min(int(clip.get('end_s', total / 30) * 30), total)
    sample = np.arange(start_f, end_f, int(30 / TRAJ_HZ))
    times = (sample - start_f) / 30.0
    rotation, origin = gravity_frame(camera[start_f])

    points_world = hands[sample, :, :, :3, 3]  # (N, 2, 21, 3)
    present = confidence[sample].mean(-1) > 0.5
    keypoints = to_frame(points_world, rotation, origin)
    wrist_rot = np.einsum('ij,nhjk->nhik', rotation, hands[sample, :, 0, :3, :3])
    wrist_rotvec = Rotation.from_matrix(wrist_rot.reshape(-1, 3, 3)).as_rotvec().reshape(len(sample), 2, 3)
    wrist_pos = keypoints[:, :, 0]
    speed = np.zeros(wrist_pos.shape[:2])
    speed[1:] = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=-1) * TRAJ_HZ
    aperture = np.linalg.norm(keypoints[:, :, 4] - keypoints[:, :, 8], axis=-1)

    contact_full, force_full, rlhnd_path = rlhnd_contact(h5_path, total)
    contact = contact_full[sample] if contact_full is not None else None
    force = force_full[sample] if force_full is not None else None
    segments = phases(times, speed, contact, present)

    retarget, fit = np.zeros((len(sample), 2, 16)), np.full((len(sample), 2), np.nan)
    for slot, side in enumerate(SIDES):
        canonical = np.stack([hand_to_canonical(p, is_left=(side == 'left')) for p in points_world[:, slot]])
        if mirror:
            canonical[..., 0] *= -1
        retarget[:, slot], fit[:, slot] = geort[side](canonical)
    retarget_valid = present.copy()

    name = f"{h5_path.parent.name}_{h5_path.stem}"
    demo = out_dir / name
    demo.mkdir(parents=True)

    def labels(t):
        out = {}
        for side in SIDES:
            out[side] = next((s['phase'] for s in segments[side] if s['start_s'] <= t < s['end_s']), 'absent')
        return out

    write_video(h5_path.with_suffix('.mp4'), start_f / 30, end_f / 30, demo / 'video.mp4', demo / 'frames', labels)
    np.savez_compressed(demo / 'hand_trajectory.npz', t=times.astype(np.float32), present=present,
                        wrist_pos=wrist_pos.astype(np.float32), wrist_rotvec=wrist_rotvec.astype(np.float32),
                        keypoints=keypoints.astype(np.float32), finger_flexion=flexion(keypoints).astype(np.float32),
                        aperture=aperture.astype(np.float32),
                        **({'contact': contact.astype(np.float32)} if contact is not None else {}),
                        **({'force': force.astype(np.float32)} if force is not None else {}))
    np.savez_compressed(demo / 'allegro_retarget.npz', q=retarget.astype(np.float32), valid=retarget_valid)
    (demo / 'phases.json').write_text(json.dumps(
        {'hands': segments, 'source': 'contact (RLHND)' if contact is not None else 'wrist speed only',
         'threshold': 0.5}, indent=2))
    which = str(attrs.get('which_llm_description', '1'))
    episode = {
        'source': 'EgoDex (Apple Vision Pro head camera, ARKit hand tracking)',
        'clip': f"{h5_path.parent.name}/{h5_path.stem}", 'similar_to_task': task,
        'person_instruction': attrs.get('llm_description2' if which == '2' else 'llm_description', ''),
        'objects': [str(o) for o in (attrs.get('llm_objects') or [])],
        'acting_hands': clip.get('hands', 'right'), 'span_s': [round(start_f / 30, 2), round(end_f / 30, 2)],
        'matching_span_note': clip.get('note', 'the whole clip'),
        'duration_s': round((end_f - start_f) / 30, 2), 'trajectory_hz': TRAJ_HZ,
        'frame': 'gravity_head0',
        'frame_note': 'metres; x right, y forward (horizontal head direction at the first frame), z up; '
                      'origin at the head at the first frame. Only relative motion is meaningful for the robot.',
        'hand_slots': ['left', 'right'],
        'keypoint_order': 'wrist, then thumb, index, middle, ring, little: 4 joints each from base to tip',
        'contact_source': 'RLHND tactile hand tracker (estimated)' if contact is not None else None,
        'retarget': 'DexJoCo teleoperation GeoRT (fingertip retargeting) per hand; dexjoco_step hand order',
    }
    (demo / 'episode.json').write_text(json.dumps(episode, indent=2))

    verify_dir.mkdir(parents=True, exist_ok=True)
    world_to_cam = np.linalg.inv(camera[sample])
    cam_points = np.einsum('nij,nhkj->nhki', world_to_cam[:, :3, :3], points_world) + world_to_cam[:, None, None, :3, 3]
    overlay_video(h5_path.with_suffix('.mp4'), start_f / 30, end_f / 30, verify_dir / f'{name}_overlay.mp4',
                  cam_points, intrinsic, TRAJ_HZ)
    return name, {'fit_error_m': {side: float(np.nanmean(fit[present[:, s], s])) if present[:, s].any() else None
                                  for s, side in enumerate(SIDES)},
                  'rlhnd': rlhnd_path, 'samples': len(sample)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='dexjoco-human root (task dirs go below)')
    parser.add_argument('--verify', type=Path, required=True)
    parser.add_argument('--geort', type=Path, required=True, help='DexJoCo teleoperation/GeoRT directory')
    parser.add_argument('--mirror', action='store_true', help='flip canonical x before GeoRT (handedness check)')
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    task = selection['task']
    assert len(selection['clips']) == 3, 'three clips per task'
    geort = {side: GeoRT(args.geort, side) for side in SIDES}
    task_dir = args.out / task
    staging = args.out / f'.{task}.staging'
    shutil.rmtree(staging, ignore_errors=True)
    names, report = [], {}
    for clip in selection['clips']:
        name, info = build_clip(clip, task, staging / 'experts', args.verify / task, geort, args.mirror)
        names.append(name)
        report[name] = info
    (staging / 'manifest.json').write_text(json.dumps({'schema': SCHEMA, 'task': task, 'experts': names}, indent=2))
    if task_dir.exists():
        shutil.rmtree(task_dir)
    staging.rename(task_dir)
    (args.verify / task / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
