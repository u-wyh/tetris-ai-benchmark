#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${TETRIS_TRANSACTION_RUN_DIR:-$root/runs/ppo_transaction_test_seed42}"
session="tetris-transaction"
command="${1:-status}"

session_running() {
  tmux has-session -t "$session" 2>/dev/null
}

case "$command" in
  start|resume)
    if session_running; then
      echo "Transactional training is already running in tmux: $session" >&2
      exit 1
    fi
    if [[ "$command" == start && -e "$run_dir/transaction_state.json" ]]; then
      echo "Run already exists; use resume." >&2
      exit 1
    fi
    if [[ "$command" == resume && ! -f "$run_dir/transaction_state.json" ]]; then
      echo "No transaction_state.json to resume." >&2
      exit 1
    fi
    mkdir -p "$run_dir/logs"
    resume_flag=""
    if [[ "$command" == resume ]]; then resume_flag="--resume"; fi
    device_flag=""
    if [[ "$command" == start && -n "${TETRIS_TRANSACTION_DEVICE:-}" ]]; then
      case "$TETRIS_TRANSACTION_DEVICE" in
        cpu|cuda) device_flag="--device $TETRIS_TRANSACTION_DEVICE" ;;
        *) echo "TETRIS_TRANSACTION_DEVICE must be cpu or cuda." >&2; exit 2 ;;
      esac
    fi
    tmux new-session -d -s "$session" \
      "cd '$root' && exec '$root/.venv/bin/python' -m training.train_transaction --run-dir '$run_dir' $resume_flag $device_flag >> '$run_dir/logs/console.log' 2>&1"
    echo "Started in tmux session $session. The process continues after this shell exits."
    ;;
  status)
    cd "$root"
    "$root/.venv/bin/python" -m training.train_transaction --run-dir "$run_dir" --status
    ;;
  *)
    echo "Usage: $0 {start|resume|status}" >&2
    exit 2
    ;;
esac
