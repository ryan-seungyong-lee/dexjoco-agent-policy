# Sourced by every Slurm job: where the experiment's checkout and runtime live.
export RAX_EXP_ROOT="$HOME/rax_haiku"
export RAX_RUNTIME_ROOT="$RAX_EXP_ROOT/runtime"
export PATH="$HOME/.local/bin:$PATH"
# The login session's /run/user/<uid> does not exist on compute nodes.
unset XDG_RUNTIME_DIR
# RAX_REPO: the frozen release an experiment job runs (exp/launch.py); default the working checkout.
cd "${RAX_REPO:-$RAX_EXP_ROOT/rax}"
source scripts/host_env.sh
# Simulator GPU: the one Slurm gave this job, by UUID (indices are renumbered inside the cgroup).
sim_gpu_uuid() { nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1; }
