#!/bin/bash
# One-time environment build on a compute node (the login node must not run installs or agents).
#SBATCH --job-name=rax-setup
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm-logs/rax-setup-%j.log
set -euo pipefail
source "$HOME/rax_haiku/rax/exp/slurm/env.sh"
bash scripts/bootstrap_sources.sh
bash scripts/setup_host_env.sh
# Claude Code's sandbox needs socat; not installed system-wide on Yangjae.
[[ -x "$RAX_HOST_ENV_ROOT/bin/socat" ]] || "$RAX_RUNTIME_ROOT/miniforge3/bin/conda" install -y -p "$RAX_HOST_ENV_ROOT" -c conda-forge socat
# The bundle sandbox passes --tmpfs a --size (bubblewrap >= 0.10); the nodes ship 0.9.
[[ -x "$RAX_HOST_ENV_ROOT/bin/bwrap" ]] || "$RAX_RUNTIME_ROOT/miniforge3/bin/conda" install -y -p "$RAX_HOST_ENV_ROOT" -c conda-forge "bubblewrap>=0.10"
which bwrap; bwrap --version
bwrap --ro-bind / / --unshare-all --die-with-parent --proc /proc --dev /dev --size 1048576 --tmpfs /tmp true && echo BWRAP_SIZE_OK
bash scripts/setup_dexjoco_env.sh
# Claude Code (native installer; version tested by RAX).
if ! command -v claude >/dev/null; then
    curl -fsSL https://claude.ai/install.sh | bash -s 2.1.296
fi
claude --version
bash scripts/doctor.sh || true
echo SETUP_CPU_DONE
