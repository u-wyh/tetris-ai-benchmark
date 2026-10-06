#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${TETRIS_RUN_DIR:-$root/runs/ppo_smoke_seed42}"
session="tetris-train"
command="${1:-status}"

session_running() {
  tmux has-session -t "$session" 2>/dev/null
}

case "$command" in
  start|resume)
    if session_running; then
      echo "Training tmux session already exists: $session" >&2
      exit 1
    fi
    if [[ "$command" == start && -e "$run_dir/config.json" ]]; then
      echo "Run already exists; use resume." >&2
      exit 1
    fi
    if [[ "$command" == resume && ! -f "$run_dir/checkpoints/latest.zip" ]]; then
      echo "Cannot resume: latest.zip is missing." >&2
      exit 1
    fi
    mkdir -p "$run_dir/logs"
    rm -f "$run_dir/PAUSE_REQUESTED"
    resume_flag=""
    if [[ "$command" == resume ]]; then resume_flag="--resume"; fi
    tmux new-session -d -s "$session" \
      "cd '$root' && exec '$root/.venv/bin/python' -m training.train_ppo --run-dir '$run_dir' $resume_flag >> '$run_dir/logs/console.log' 2>&1"
    echo "Started in tmux session $session. Use: $root/scripts/train_control.sh status"
    ;;
  pause)
    if ! session_running; then
      echo "No running training session to pause." >&2
      exit 1
    fi
    touch "$run_dir/PAUSE_REQUESTED"
    echo "Pause requested; the trainer will finish its current chunk and save latest.zip."
    ;;
  status)
    cd "$root"
    "$root/.venv/bin/python" -m training.train_ppo --run-dir "$run_dir" --status
    ;;
  *)
    echo "Usage: $0 {start|status|pause|resume}" >&2
    exit 2
    ;;
esac
