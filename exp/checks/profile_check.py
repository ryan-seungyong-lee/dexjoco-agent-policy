#!/usr/bin/env python3
"""GPU check that each observation profile gives the agent exactly its inputs, in the real
DexJoCo simulator: prepares a throwaway session per profile, starts one exploration episode,
observes once and reports the keys and attachment kinds the agent would receive.

  python exp/checks/profile_check.py --gpu <uuid> --task water_plant
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
from services.mcp_contract import PROFILES  # noqa: E402
from services.dexjoco.backend import DexjocoBackend  # noqa: E402

EXPECTED = {'rgbd': (True, True), 'rgb-depth': (True, False), 'rgb-camera': (False, True), 'rgb-only': (False, False)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--task', default='water_plant')
    args = parser.parse_args()
    report, ok = {}, True
    for profile in PROFILES:
        config_path = subprocess.check_output(
            [sys.executable, 'scripts/configure_auto_research.py', '--task', args.task, '--sim-gpu', args.gpu,
             '--episodes', '1', '--agent-cli', 'claude', '--agent-provider', 'anthropic',
             '--external-perception', 'off', '--observation-profile', profile, '--demonstration-context', 'none'],
            text=True, cwd=PROJECT).strip().splitlines()[-1]
        config = with_task_timeouts(Configuration.from_dict(json.loads(Path(config_path).read_text())))
        backend = DexjocoBackend(config)
        try:
            backend.start(formal=False, seed=config.exploration_seeds[0])
            packet, content = backend.call('dexjoco_observe', {})
            kinds = [a['kind'] for a in packet.get('attachments', [])]
            got_depth = 'depth' in kinds or 'depth' in packet
            got_camera = 'camera_intrinsics' in packet or 'camera_poses' in packet
            tools = sorted(t['name'] for t in backend.tools())
            report[profile] = {'keys': sorted(packet), 'attachment_kinds': kinds, 'images_returned': len(content),
                               'tools': tools, 'depth': got_depth, 'camera_params': got_camera}
            good = (got_depth, got_camera) == EXPECTED[profile] and \
                ('dexjoco_pixel_to_position' in tools) == (profile == 'rgbd')
            report[profile]['ok'] = good
            ok &= good
        finally:
            backend.close()
            shutil.rmtree(Path(config_path).parent, ignore_errors=True)
    print(json.dumps(report, indent=2))
    print('PROFILE_CHECK', 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
