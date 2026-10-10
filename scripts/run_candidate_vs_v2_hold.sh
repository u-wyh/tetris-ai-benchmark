#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
run_dir="$root/runs/ppo_candidate_vs_v2_hold_final100"
mkdir -p "$run_dir"
exec >> "$run_dir/console.log" 2>&1

"$root/.venv/bin/python" -m training.evaluation.candidate_vs_v2_hold

paths=(CHANGELOG.md reports/experiments/ppo_candidate_vs_v2_hold_final100.md)
if [[ -n "$(git status --porcelain -- "${paths[@]}")" ]]; then
  git add -- "${paths[@]}"
  git commit --only -m 'docs(evaluation): report Candidate versus V2 Hold-only formal test' -- "${paths[@]}"
fi
git push origin main
