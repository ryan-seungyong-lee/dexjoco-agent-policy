#!/usr/bin/env python3
"""Frame-free sanity check of the human->Allegro GeoRT retarget, normal vs mirrored canonical frame.

Raw tip error mixes in the offset between the human wrist frame and the robot hand frame, so this
also reports the error after removing each clip's mean offset and, most tellingly, how well the
robot's thumb-to-finger distances follow the human's over time (pinch correlation).
  python exp/human_demos/geort_check.py --demos <out>/fold_glasses/experts --geort <GeoRT dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_human_demos import GeoRT, SIDES, hand_to_canonical  # noqa: E402


def pinch(tips, thumb):
    return np.stack([np.linalg.norm(tips[:, i] - tips[:, thumb], axis=-1)
                     for i in range(tips.shape[1]) if i != thumb], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demos', type=Path, required=True)
    parser.add_argument('--geort', type=Path, required=True)
    args = parser.parse_args()
    models = {side: GeoRT(args.geort, side) for side in SIDES}
    report = {}
    for demo in sorted(p for p in args.demos.iterdir() if p.is_dir()):
        data = np.load(demo / 'hand_trajectory.npz')
        for slot, side in enumerate(SIDES):
            present = data['present'][:, slot]
            if present.sum() < 10:
                continue
            model = models[side]
            ids = model.human_ids
            thumb = ids.index(4)
            for mirror in (False, True):
                canonical = np.stack([hand_to_canonical(p.copy(), side == 'left')
                                      for p in data['keypoints'][present, slot]])
                if mirror:
                    canonical[..., 0] *= -1
                import torch
                with torch.no_grad():
                    reached = model.fk(model.ik(torch.from_numpy(canonical[:, ids].astype(np.float32)))).numpy()
                human = canonical[:, ids]
                raw = np.linalg.norm(reached - human, axis=-1).mean()
                offset = (reached - human).mean(0)
                centered = np.linalg.norm(reached - human - offset, axis=-1).mean()
                hp, rp = pinch(human, thumb), pinch(reached, thumb)
                corr = [float(np.corrcoef(hp[:, j], rp[:, j])[0, 1]) for j in range(hp.shape[1])]
                report[f'{demo.name}/{side}/{"mirror" if mirror else "normal"}'] = {
                    'frames': int(present.sum()), 'raw_m': round(float(raw), 4),
                    'centered_m': round(float(centered), 4), 'offset_m': np.round(offset.mean(0), 3).tolist(),
                    'pinch_corr': [round(c, 3) for c in corr],
                    'human_pinch_range_m': np.round(np.ptp(hp, 0), 3).tolist(),
                    'robot_pinch_range_m': np.round(np.ptp(rp, 0), 3).tolist()}
    for key, value in report.items():
        print(key, json.dumps(value))
    for mode in ('normal', 'mirror'):
        rows = [v for k, v in report.items() if k.endswith(mode)]
        print('SUMMARY', mode, 'centered_m', round(np.mean([r['centered_m'] for r in rows]), 4),
              'pinch_corr', round(float(np.nanmean([c for r in rows for c in r['pinch_corr']])), 3))


if __name__ == '__main__':
    main()
