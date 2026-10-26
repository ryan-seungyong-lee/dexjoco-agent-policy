#!/usr/bin/env python3
"""Per-arm results so far: formal episodes done, successes, time and tokens (both modes).
  python3 exp/tally.py [--root ~/rax_haiku]"""
import argparse
import glob
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.home() / 'rax_haiku')
    args = parser.parse_args()
    for arm in sorted((args.root / 'arms').iterdir()):
        config = arm / 'config_path'
        if '.' in arm.name or not config.is_file():
            continue
        session = Path(config.read_text().strip()).parent
        manifest = json.loads((session / 'exp_manifest.json').read_text()) if (session / 'exp_manifest.json').is_file() else {}
        runs = [json.loads(Path(p).read_text()) for p in glob.glob(str(session / 'published/runs/formal-*/result.json'))]
        # An episode whose agent hit the subscription's usage limit measured nothing: left out.
        invalid = [r for r in runs if 'limit' in json.dumps(r.get('agent_errors') or [])]
        runs = [r for r in runs if r not in invalid]
        scores = session / 'formal-scores.json'
        done = len(runs)
        success = sum(bool(r.get('task_complete')) for r in runs)
        errors = sum(r.get('reason') not in ('exit', 'episode_ended') or bool(r.get('agent_errors')) for r in runs)
        minutes = [r['timing']['total_seconds'] / 60 for r in runs if r.get('timing')]
        usage = [r.get('agent_token_usage') or {} for r in runs]
        out = sum(u.get('output_tokens', 0) for u in usage)
        total = sum(u.get('total_tokens', 0) for u in usage)
        target = manifest.get('formal_episodes', '?')
        line = (f"{arm.name:46s} {done:3d}/{target} success {success:3d}"
                f" ({100 * success / done if done else 0:5.1f}%) err {errors}"
                + (f" usage-limited {len(invalid)}" if invalid else ""))
        if minutes:
            line += f" | {sum(minutes) / len(minutes):5.1f} min/ep"
        if total:
            line += f" | out {out / 1e3:6.0f}k tot {total / 1e6:6.1f}M tok"
        if scores.is_file():
            line += f" | formal-scores {json.loads(scores.read_text()).get('average_score')}"
        print(line)


if __name__ == '__main__':
    main()
