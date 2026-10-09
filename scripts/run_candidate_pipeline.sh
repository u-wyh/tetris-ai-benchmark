#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/ppo_candidate_pipeline_seed42
exec .venv/bin/python -m training.evaluation.candidate_pipeline >> runs/ppo_candidate_pipeline_seed42/console.log 2>&1
