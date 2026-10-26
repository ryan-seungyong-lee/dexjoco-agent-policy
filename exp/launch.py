#!/usr/bin/env python3
"""Submit one phase of exp/arms.yaml to Yangjae Slurm from the laptop, one job per arm.

Every arm runs a frozen release of the committed code (~/rax_haiku/releases/<sha>, made on first
use), so syncing the working checkout never changes a session that is already running.

An arm that already has a session (arms/<id>/config_path on the cluster) or a queued/running
job carrying its id is skipped, so rerunning the command never duplicates work. At most --slots
arms run at once: each job is named rax-slot<k> with --dependency=singleton, so Slurm runs one
job per name at a time; the arm id rides in the job comment.

  python3 exp/launch.py --phase A [--dry-run] [--qos background] [--slots 8]
"""
import argparse
import os
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

HOST = os.environ.get('RAX_SLURM_HOST') or sys.exit('set RAX_SLURM_HOST to the ssh alias of the Slurm login node')
REMOTE_ROOT = os.environ.get('RAX_REMOTE_ROOT', '~/rax_haiku')  # RAX_EXP_ROOT on the cluster
PROJECT = Path(__file__).resolve().parents[1]


def arms(phase_name):
    spec = yaml.safe_load((PROJECT / 'exp' / 'arms.yaml').read_text())
    phase = spec['phases'][phase_name]
    keys = ('episodes', 'model', 'effort', 'mode', 'formal_episodes', 'agent', 'codex_profile')
    defaults = {'agent': 'claude', **spec['defaults'], **{k: phase[k] for k in keys if k in phase}}
    for task in phase.get('tasks', spec['tasks']):
        for rep in phase.get('reps', spec['reps']):
            for arm in phase['arms']:
                settings = {**defaults, **{k: arm[k] for k in keys if k in arm}}  # an arm may override them
                arm_id = f"{phase_name}_{task}_{arm['profile']}_{arm['expert']}_r{rep}"
                yield {'ARM_ID': arm_id, 'TASK': task, 'PROFILE': arm['profile'], 'EXPERT': arm['expert'],
                       'REP': str(rep), 'EPISODES': str(settings['episodes']), 'MODEL': settings['model'],
                       'EFFORT': settings['effort'], 'MODE': settings['mode'], 'AGENT': settings['agent'],
                       'FORMAL_EPISODES': str(settings['formal_episodes']),
                       **({'CODEX_PROFILE': settings['codex_profile']} if settings.get('codex_profile') else {})}


def ssh(command):
    return subprocess.run(['ssh', HOST, command], check=True, capture_output=True, text=True).stdout


def release():
    """The remote directory of a frozen copy of HEAD (uncommitted changes refuse to launch)."""
    if subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=PROJECT,
                      capture_output=True, text=True, check=True).stdout.strip():
        raise SystemExit('Commit your changes first: every arm runs a frozen release of HEAD')
    sha = subprocess.run(['git', 'rev-parse', '--short=12', 'HEAD'], cwd=PROJECT, capture_output=True, text=True,
                         check=True).stdout.strip()
    path = f'{REMOTE_ROOT}/releases/{sha}'
    if ssh(f'test -f {path}/.complete && echo yes || true').strip() != 'yes':
        ssh(f'mkdir -p {path}')
        subprocess.run(['rsync', '-az', '--exclude', 'runtime', '--exclude', 'DexJoCo', '--exclude', '__pycache__',
                        '--exclude', '.local-runtime', f'{PROJECT}/', f'{HOST}:{path.replace("~/", "")}/'], check=True)
        ssh(f'cd {path} && ln -sfn $(readlink -f {REMOTE_ROOT}/rax/DexJoCo) DexJoCo && '
            f'ln -sfn $(readlink -f {REMOTE_ROOT}/rax/runtime) runtime && touch .complete')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--qos', choices=('base', 'background'), default='background',
                        help='background: preemptible extra quota (jobs requeue and resume their session)')
    # Every live copy is a Claude session on the one subscription: 3 arms x 3 copies stay under its limit.
    parser.add_argument('--slots', type=int, default=3, help='arms running at once')
    parser.add_argument('--after', help='job id the arms with demonstrations wait for (afterok), e.g. their demo setup')
    args = parser.parse_args()
    home = ssh('echo $HOME').strip()  # The job environment gets an absolute path.
    repo = (release() if not args.dry_run else f'{REMOTE_ROOT}/releases/<sha>').replace('~', home, 1)
    print('release', repo)
    started = set(ssh(f'ls {REMOTE_ROOT}/arms 2>/dev/null || true').split())
    jobs = [line.split('|') for line in ssh("squeue -u $USER -h -o '%j|%k'").split()]
    queued = {comment for name, comment in jobs if name.startswith('rax-slot')}
    load = {k: sum(name == f'rax-slot{k}' for name, _ in jobs) for k in range(args.slots)}
    for arm in arms(args.phase):
        if arm['ARM_ID'] in started or arm['ARM_ID'] in queued:
            print('skip', arm['ARM_ID'], '(started)' if arm['ARM_ID'] in started else '(queued)')
            continue
        export = 'ALL,' + ','.join(f'{k}={v}' for k, v in {**arm, 'RAX_REPO': repo}.items())
        qos = '--qos=background --requeue ' if args.qos == 'background' else ''
        slot = min(load, key=load.get)
        load[slot] += 1
        dependency = 'singleton' + (f',afterok:{args.after}' if args.after and arm['EXPERT'] != 'none' else '')
        command = (f'cd {repo} && mkdir -p ~/slurm-logs && sbatch {qos}--job-name=rax-slot{slot} '
                   f'--output=$HOME/slurm-logs/rax-arm-%j.log --dependency={dependency} '
                   f'--comment={shlex.quote(arm["ARM_ID"])} '
                   f'--export={shlex.quote(export)} exp/slurm/session.sbatch')
        if args.dry_run:
            print(command)
        else:
            print(arm['ARM_ID'], ssh(command).strip())


if __name__ == '__main__':
    main()
