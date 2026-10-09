#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
run_dir="$root/runs/ppo_candidate_raw_10m_seed42"
mkdir -p "$run_dir/logs"
exec >> "$run_dir/logs/continuation-console.log" 2>&1

"$root/.venv/bin/python" -m training.train_vector_transaction --run-dir "$run_dir" --resume
"$root/.venv/bin/python" -m training.evaluation.candidate_10m_report --candidate-run "$run_dir"
