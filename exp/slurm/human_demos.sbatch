#!/bin/bash
# Human demo data: rank EgoDex candidates and (with SELECTION) build the demos. MODE=candidates|build|test
# Data: EGODEX_ROOT (EgoDex train set) and RLHND_ROOT (matching hand reconstructions), or EGODEX_SUBSET_REPO,
# a Hugging Face dataset holding egodex/ and rlhnd/ that is downloaded once.
#SBATCH --job-name=rax-human
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm-logs/rax-human-%j.log
set -euo pipefail
source "${RAX_REPO:-$HOME/rax_haiku/rax}/exp/slurm/env.sh"
DATA="$RAX_EXP_ROOT/data/egodex_subset"
if [[ -z "${EGODEX_ROOT:-}" ]]; then
    if [[ ! -f "$DATA/.complete" ]]; then
        # host_env.sh moves HF_HOME into the runtime; the login token stays in the default cache.
        HF_TOKEN_PATH="$HOME/.cache/huggingface/token" hf download "${EGODEX_SUBSET_REPO:?set EGODEX_ROOT or EGODEX_SUBSET_REPO}" \
            --repo-type dataset --local-dir "$DATA"
        touch "$DATA/.complete"
    fi
    export EGODEX_ROOT="$DATA/egodex" RLHND_ROOT="$DATA/rlhnd"
fi
OUT="$RAX_EXP_ROOT/human_demos"
GEORT="$RAX_EXP_ROOT/rax/DexJoCo/teleoperation/GeoRT"
case "${MODE:-test}" in
  candidates)
    "$RAX_PYTHON" exp/human_demos/human_candidates.py --out "$OUT/candidates" --per-task 30 ;;
  test)
    for mirror in "" 1; do
        "$RAX_PYTHON" exp/human_demos/build_human_demos.py --selection exp/human_demos/selections/test_fold_glasses.json \
            --out "$OUT/test${mirror:+_mirror}" --verify "$OUT/verify${mirror:+_mirror}" --geort "$GEORT" ${mirror:+--mirror}
    done ;;
  build)
    : "${SELECTION:?}"
    "$RAX_PYTHON" exp/human_demos/build_human_demos.py --selection "$SELECTION" \
        --out "$RAX_RUNTIME_ROOT/reference-demos/dexjoco-human" --verify "$OUT/verify" --geort "$GEORT" ${MIRROR:+--mirror} ;;
esac
echo HUMAN_${MODE:-test}_DONE
