#!/usr/bin/env python3
"""GPU check of agent as policy's target-level motion in the real DexJoCo simulator: an
--agent-policy session's backend exposes dexjoco_move_ee / dexjoco_hand (not dexjoco_step), a
move reaches its target, a hand close is tracked, and a later move keeps the hand closed.

  python exp/checks/target_motion_check.py --gpu <uuid> --task click_mouse
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from services.controller.config import Configuration, with_task_timeouts  # noqa: E402
from services.dexjoco.backend import DexjocoBackend  # noqa: E402

CLOSED = [0.0, 0.9, 0.9, 0.9] * 3 + [1.2, 0.6, 0.8, 0.8]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--task', default='click_mouse')
    args = parser.parse_args()
    config_path = subprocess.check_output(
        [sys.executable, 'scripts/configure_auto_research.py', '--task', args.task, '--sim-gpu', args.gpu,
         '--episodes', '1', '--formal-episodes', '1', '--agent-cli', 'claude', '--agent-provider', 'anthropic',
         '--external-perception', 'off', '--demonstration-context', 'none', '--agent-policy', 'on'],
        text=True, cwd=PROJECT).strip().splitlines()[-1]
    config = with_task_timeouts(Configuration.from_dict(json.loads(Path(config_path).read_text())))
    backend = DexjocoBackend(config)
    report, ok = {}, True
    try:
        tools = sorted(t['name'] for t in backend.tools())
        report['tools'] = tools
        ok &= 'dexjoco_move_ee' in tools and 'dexjoco_hand' in tools and 'dexjoco_step' not in tools
        backend.start(formal=False, seed=config.exploration_seeds[0])
        moves = {}
        for name, arguments in [('down_10cm', {'delta_m': [0.0, 0.0, -0.10]}),
                                ('forward_precise', {'delta_m': [0.05, 0.0, 0.0], 'speed': 'precision',
                                                     'position_tolerance_m': 0.005}),
                                ('side_20cm', {'delta_m': [0.0, 0.2, 0.0]}),
                                ('close_hand', {'joints_rad': CLOSED, 'steps': 20}),
                                ('up_keeps_hand', {'delta_m': [0.0, 0.0, 0.08]}),
                                ('into_table', {'delta_m': [0.0, 0.0, -0.6], 'max_settle_steps': 120})]:
            tool = 'dexjoco_hand' if name == 'close_hand' else 'dexjoco_move_ee'
            packet, content = backend.call(tool, arguments)
            motion = packet['motion']
            moves[name] = {k: motion.get(k) for k in ('reached', 'position_error_m', 'rotation_error_rad', 'steps_used', 'settle_lead_m', 'stalled',
                                                      'interpolated_steps', 'settle_steps', 'blocked_joints')}
            moves[name]['step_id'], moves[name]['images'] = packet['step_id'], len(content)
            if name == 'into_table':
                ok &= bool(motion['stalled']) and motion['settle_steps'] < 60
            elif tool == 'dexjoco_move_ee':
                ok &= bool(motion['reached'])
            hand = packet['state'][7:23]
            moves[name]['hand_mean_rad'] = round(sum(hand) / 16, 3)
        ok &= moves['up_keeps_hand']['hand_mean_rad'] > 0.5  # the closed hand stayed closed
        report['moves'] = moves
    finally:
        backend.close()
        shutil.rmtree(Path(config_path).parent, ignore_errors=True)
    print(json.dumps(report, indent=2))
    print('TARGET_MOTION_CHECK', 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
