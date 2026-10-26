#!/bin/bash
#SBATCH --job-name=rax-target-check
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=0:30:00
#SBATCH --output=slurm-logs/rax-target-check-%j.log
set -euo pipefail
source "$HOME/rax_haiku/rax/exp/slurm/env.sh"
for task in click_mouse; do
    PYTHONPATH=. "$RAX_PYTHON" exp/checks/target_motion_check.py --gpu "$(sim_gpu_uuid)" --task "$task"
done
