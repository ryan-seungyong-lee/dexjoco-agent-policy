#!/bin/bash
# One experiment arm as a Slurm job on Yangjae (exp/launch.py submits these; see exp/run_arm.sh
# for the variables). A preempted job requeues and resumes its session.
#SBATCH --job-name=rax-arm
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/rax-arm-%j.log
set -euo pipefail
source "${RAX_REPO:-$HOME/rax_haiku/rax}/exp/slurm/env.sh"
SIM_GPU="$(sim_gpu_uuid)" exec bash exp/run_arm.sh
