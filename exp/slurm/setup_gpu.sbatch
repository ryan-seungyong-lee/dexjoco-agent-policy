#!/bin/bash
# Expert demos for the experiment's tasks (replayed in the official env: needs EGL), their keyposes.json
# summaries (exp/demos/robot_keyposes.py) and the doctor. KEYPOSES_ONLY=1: only (re)write the summaries.
#SBATCH --job-name=rax-setup-gpu
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=32G
#SBATCH --time=3:00:00
#SBATCH --output=slurm-logs/rax-setup-gpu-%j.log
set -euo pipefail
source "${RAX_REPO:-$HOME/rax_haiku/rax}/exp/slurm/env.sh"
GPU="$(sim_gpu_uuid)"; nvidia-smi -L
TASKS=(${TASKS:-water_plant fold_glasses bimanual_microwave_cook})
for task in "${TASKS[@]}"; do
    [[ -n "${KEYPOSES_ONLY:-}" ]] && break
    "$RAX_PYTHON" scripts/setup_dexjoco_demos.py --gpu "$GPU" --task "$task" --candidates 12
done
"$RAX_PYTHON" exp/demos/robot_keyposes.py --gpu "$GPU" $(printf -- '--task %s ' "${TASKS[@]}")
[[ -n "${KEYPOSES_ONLY:-}" ]] && { echo SETUP_GPU_DONE; exit 0; }
"$RAX_PYTHON" scripts/dexjoco_doctor.py --gpu "$GPU" $(printf -- '--task %s ' "${TASKS[@]}")
echo SETUP_GPU_DONE
