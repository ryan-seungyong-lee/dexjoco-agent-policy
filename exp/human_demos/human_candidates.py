#!/usr/bin/env python3
"""Rank human clips (EgoDex) as in-context demonstrations for the experiment's DexJoCo tasks.

Reads only the hdf5 attributes and hand-joint confidences of each clip, keeps clips whose
description or objects match the task's keywords, and writes candidates.jsonl plus a contact
sheet (8 frames) per top candidate for a person to pick from.

Usage (a CPU job; EGODEX_ROOT is the EgoDex train set):
  python exp/human_demos/human_candidates.py --out <dir>/human_candidates [--per-task 30]
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import h5py
import numpy as np

import os
EGODEX = Path(os.environ.get('EGODEX_ROOT', 'data/egodex-train'))
# DexJoCo task -> (EgoDex task folders, keyword pattern on description/objects, hands the task uses)
TASKS = {
    'fold_glasses': (['part4/put_in_take_out_glasses'], r'fold|put .*(in|into|back)|place', 'right'),
    'water_plant': (['part4/pour', 'part1/clean_surface', 'part5/wipe_kitchen_surfaces', 'part1/clean_cups'],
                    r'spray|water(ing)? ?can|plant|pour .*(water|plant)', 'right'),
    'bimanual_microwave_cook': (['part4/stock_unstock_fridge', 'part3/open_close_insert_remove_box',
                                 'part1/boil_serve_egg'],
                                r'microwave|oven|fridge|refrigerator|door|put .* (in|into)', 'both'),
}
HAND_CONF = ('leftHand', 'rightHand')


def clip_info(h5_path):
    with h5py.File(h5_path, 'r') as f:
        attrs = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in f.attrs.items()}
        # Some clips carry no confidences: count their hands as tracked.
        conf = ({name: float(np.mean(f['confidences'][name][()] > 0.5)) for name in HAND_CONF
                 if name in f['confidences']} if 'confidences' in f else {name: 1.0 for name in HAND_CONF})
        frames = int(f['transforms']['camera'].shape[0])
    which = str(attrs.get('which_llm_description', '1'))
    description = attrs.get('llm_description2' if which == '2' else 'llm_description', '')
    return {'path': str(h5_path), 'video': str(h5_path.with_suffix('.mp4')), 'frames': frames,
            'seconds': frames / 30.0, 'description': description,
            'descriptions_all': [attrs.get(k, '') for k in ('description1', 'description2')],
            'objects': [str(o) for o in (attrs.get('llm_objects') or [])],
            'environment': attrs.get('environment', ''), 'hand_presence': conf}


def score(info, pattern, hands):
    text = ' '.join([info['description'], *info['objects']]).lower()
    if not re.search(pattern, text):
        return None
    presence = info['hand_presence']
    used = [presence.get('rightHand', 0)] if hands == 'right' else [presence.get('leftHand', 0),
                                                                      presence.get('rightHand', 0)]
    duration = info['seconds']
    duration_fit = 1.0 if 3 <= duration <= 25 else 0.3
    return float(min(used)) * duration_fit


def contact_sheet(video, out, n=8, width=320):
    capture = cv2.VideoCapture(video)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for index in np.linspace(0, max(total - 1, 0), n).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            frame = np.zeros((180, 320, 3), np.uint8)
        frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
        cv2.putText(frame, f'{index / 30:.1f}s', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(frame)
    capture.release()
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, n, 4)]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-task', type=int, default=30)
    parser.add_argument('--max-scan', type=int, default=4000, help='clips scanned per EgoDex folder')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for task, (folders, pattern, hands) in TASKS.items():
        ranked = []
        for folder in folders:
            if not (EGODEX / folder).is_dir():
                print('missing folder (not copied):', folder, flush=True)
                continue
            files = sorted((EGODEX / folder).glob('*.hdf5'), key=lambda p: int(p.stem))[:args.max_scan]
            for h5_path in files:
                try:
                    info = clip_info(h5_path)
                except Exception as error:  # damaged clip: skip, but say so
                    print('skip', h5_path, error, flush=True)
                    continue
                value = score(info, pattern, hands)
                if value is not None:
                    ranked.append({**info, 'score': value, 'folder': folder})
            print(task, folder, 'matches so far', len(ranked), flush=True)
        ranked.sort(key=lambda item: -item['score'])
        top = ranked[:args.per_task]
        sheets = args.out / task
        sheets.mkdir(exist_ok=True)
        for rank, item in enumerate(top):
            name = f"{rank:02d}_{item['folder'].split('/')[-1]}_{Path(item['path']).stem}.jpg"
            contact_sheet(item['video'], sheets / name)
            item['sheet'] = f'{task}/{name}'
        with open(args.out / f'{task}.jsonl', 'w') as stream:
            for item in top:
                stream.write(json.dumps(item) + '\n')
        print(task, 'total matches', len(ranked), 'kept', len(top), flush=True)


if __name__ == '__main__':
    main()
