#!/bin/bash
# Run one experiment arm end to end on this machine: configure a session, launch the agent, record
# the outcome. Slurm jobs (exp/slurm/session.sbatch) and plain GPU machines both use it.
#
# Required:  ARM_ID TASK PROFILE EXPERT REP
#   PROFILE  rgbd | rgb-depth | rgb-camera | rgb-only
#   EXPERT   none | full | video | human-video | human-full    (demonstrations in context)
# Optional:
#   MODE             agp (agent as policy) | cap (code as policy, default)
#   AGENT            claude (default) | codex
#   MODEL            Claude model id (default claude-sonnet-5-5); Codex takes its model from CODEX_PROFILE
#   EFFORT           xhigh (default) | high | max
#   EPISODES         exploration episodes (cap; default 5)
#   FORMAL_EPISODES  official evaluation episodes (default 150; agp pilots use 3)
#   PROVIDER, KEY    Claude login: claude-login + the setup-token file (default), or anthropic + an API key file
#   CODEX_PROFILE, CODEX_HOME   Codex provider profile (<CODEX_HOME>/<profile>.config.toml)
#   SIM_GPU          GPU UUID for the simulator (default: the first visible GPU)
#   RAX_EXP_ROOT     where arms/ and ledger.jsonl live (default ~/rax_haiku)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -n "${RAX_PYTHON:-}" ]] || source scripts/host_env.sh
: "${ARM_ID:?}" "${TASK:?}" "${PROFILE:?}" "${EXPERT:?}" "${REP:?}"
MODE="${MODE:-cap}" AGENT="${AGENT:-claude}" MODEL="${MODEL:-claude-sonnet-5-5}" EFFORT="${EFFORT:-xhigh}"
EPISODES="${EPISODES:-5}" FORMAL_EPISODES="${FORMAL_EPISODES:-150}"
RAX_EXP_ROOT="${RAX_EXP_ROOT:-$HOME/rax_haiku}"
SIM_GPU="${SIM_GPU:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)}"
# No operator answers these runs; the harness adds a note saying so to every prompt.
export RAX_UNATTENDED=1

if [[ "$AGENT" == claude ]]; then
    # claude-login: a Claude subscription's `claude setup-token` token; anthropic: a pay-per-use API key.
    PROVIDER="${PROVIDER:-claude-login}"
    KEY="${KEY:-$HOME/.config/rax/$([[ $PROVIDER == anthropic ]] && echo anthropic_key || echo claude-experiment.token)}"
    configure_agent=(--agent-cli claude --agent-provider "$PROVIDER")
    launch_agent=(--claude-model "$MODEL" --claude-key-file "$KEY")
else
    PROVIDER=codex-profile
    MODEL="codex:${CODEX_PROFILE:?set CODEX_PROFILE to a Codex provider profile}"
    configure_agent=(--agent-cli codex)
    launch_agent=(--profile "$CODEX_PROFILE" ${CODEX_HOME:+--provider-home "$CODEX_HOME"})
fi

ARM_DIR="$RAX_EXP_ROOT/arms/$ARM_ID"; mkdir -p "$ARM_DIR"
LEDGER="$RAX_EXP_ROOT/ledger.jsonl"
log() { printf '{"time":"%s","arm":"%s","job":"%s","host":"%s","event":"%s"%s}\n' \
          "$(date -u +%FT%TZ)" "$ARM_ID" "${SLURM_JOB_ID:-}" "$(hostname)" "$1" "${2:-}" >> "$LEDGER"; }

if [[ -f "$ARM_DIR/config_path" ]]; then
    # A requeued (preempted) job resumes the same session on its new GPU.
    CFG="$(cat "$ARM_DIR/config_path")"
    bash scripts/move_session_gpus.sh --config "$CFG" --sim-gpu "$SIM_GPU" || true
    log resume ",\"config\":\"$CFG\""
    bash scripts/start_auto_research_agent.sh --config "$CFG" --resume --non-interactive "${launch_agent[@]}"
else
    expert_args=(--demonstration-context none)
    [[ "$EXPERT" != none ]] && expert_args+=(--expert-demonstrations 3 --expert-content "$EXPERT")
    mode_args=(--formal-episodes "$FORMAL_EPISODES")
    [[ "$MODE" == agp ]] && mode_args+=(--agent-policy on)
    CFG="$("$RAX_PYTHON" scripts/configure_auto_research.py --task "$TASK" --sim-gpu "$SIM_GPU" \
        --episodes "$EPISODES" "${configure_agent[@]}" --agent-effort "$EFFORT" \
        --external-perception off --episode-reviews off --observation-profile "$PROFILE" \
        --exploration-start-seed $((1 + 100 * REP)) --formal-workers 4 "${expert_args[@]}" "${mode_args[@]}")"
    echo "$CFG" > "$ARM_DIR/config_path"
    ARM_ID="$ARM_ID" TASK="$TASK" PROFILE="$PROFILE" EXPERT="$EXPERT" REP="$REP" EPISODES="$EPISODES" MODE="$MODE" \
    FORMAL_EPISODES="$FORMAL_EPISODES" AGENT="$AGENT" MODEL="$MODEL" PROVIDER="$PROVIDER" EFFORT="$EFFORT" \
    SIM_GPU="$SIM_GPU" "$RAX_PYTHON" - "$(dirname "$CFG")" <<'PY'
import json, os, socket, subprocess, sys, pathlib
e = os.environ
manifest = {key.lower(): e[key] for key in ('ARM_ID', 'TASK', 'PROFILE', 'EXPERT', 'MODE', 'AGENT', 'MODEL',
                                            'PROVIDER', 'EFFORT')}
manifest.update(rep=int(e['REP']), episodes=int(e['EPISODES']), formal_episodes=int(e['FORMAL_EPISODES']),
                unattended=True, gpu=e['SIM_GPU'], host=socket.gethostname(), slurm_job=e.get('SLURM_JOB_ID', ''),
                exploration_start_seed=1 + 100 * int(e['REP']),
                git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                                       text=True).strip()))
(pathlib.Path(sys.argv[1]) / 'exp_manifest.json').write_text(json.dumps(manifest, indent=2))
PY
    log start ",\"config\":\"$CFG\""
    bash scripts/start_auto_research_agent.sh --config "$CFG" --non-interactive "${launch_agent[@]}"
fi

SESSION="$(dirname "$CFG")"
score() { "$RAX_PYTHON" -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]))" "$@"; }
if [[ -f "$SESSION/agent-formal-scores.json" ]]; then   # agp: live agents played the formal episodes
    log done ",\"score\":$(score "$SESSION/agent-formal.json" success_rate)"
elif [[ -f "$SESSION/formal-scores.json" ]]; then       # cap: the submitted program did
    log done ",\"score\":$(score "$SESSION/formal-scores.json" average_score)"
else
    log ended_without_formal
fi
