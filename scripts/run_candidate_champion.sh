#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
run_dir="$root/runs/ppo_candidate_champion"
mkdir -p "$run_dir"
exec >> "$run_dir/console.log" 2>&1

"$root/.venv/bin/python" -m training.evaluation.champion_pipeline

# Only the generated reports and changelog belong in the final commit.
paths=(CHANGELOG.md reports/experiments/ppo_candidate_champion_selection.md
       reports/experiments/ppo_candidate_final_100seed.md)
if [[ -n "$(git status --porcelain -- "${paths[@]}")" ]]; then
  git add -- "${paths[@]}"
  git commit --only -m 'docs(evaluation): report frozen Candidate champion final test' -- "${paths[@]}"
fi
git push origin main
