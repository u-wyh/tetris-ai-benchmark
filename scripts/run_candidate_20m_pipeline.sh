#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
out="$root/runs/ppo_candidate_champion_20m_selection"
mkdir -p "$out"
exec >> "$out/console.log" 2>&1

"$root/.venv/bin/python" -m training.evaluation.candidate_20m_pipeline

paths=(CHANGELOG.md reports/experiments/ppo_candidate_20m_checkpoint_selection.md
  reports/experiments/ppo_candidate_champion_v2_final.md
  reports/experiments/ppo_candidate_vs_traditional_v2.md)
if [[ -n "$(git status --porcelain -- "${paths[@]}")" ]]; then
  git add -- "${paths[@]}"
  git commit -m 'docs(evaluation): report Candidate 20M selection and V2 comparison' -- "${paths[@]}"
fi
git push origin main
