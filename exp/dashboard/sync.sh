#!/bin/bash
# Collect the cluster's sessions into dashboard data (as a short Slurm job, not on the login node),
# copy it here and serve the dashboard at http://localhost:${PORT:-8777}.
#   bash exp/dashboard/sync.sh [arm_id ...]      # no arm ids: every session
# Env: RAX_SLURM_HOST (ssh alias of the Slurm login node, required), RAX_REMOTE_ROOT (default ~/rax_haiku),
#      NO_SERVE=1 to only update the data.
set -euo pipefail
HOST="${RAX_SLURM_HOST:?set RAX_SLURM_HOST to the ssh alias of the Slurm login node}" ROOT="${RAX_REMOTE_ROOT:-~/rax_haiku}"
SITE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/site"
ARMS="${*:+--arms $*}"
ssh "$HOST" "cd $ROOT/rax && srun --quiet -p all --qos=background -c 2 --mem=8G -t 30 -J rax-dashboard \
    runtime/envs/host/bin/python exp/dashboard/collect.py --sessions runtime/auto-research --arms-dir $ROOT/arms --out $ROOT/dashboard $ARMS"
rsync -az --delete "$HOST:${ROOT#\~/}/dashboard/" "$SITE/data/"
[[ -n "${NO_SERVE:-}" ]] && exit 0
echo "Dashboard: http://localhost:${PORT:-8777}"
cd "$SITE" && exec python3 -m http.server "${PORT:-8777}" --bind 127.0.0.1
