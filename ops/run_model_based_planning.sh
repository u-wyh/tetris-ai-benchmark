#!/usr/bin/env bash
set -euo pipefail
cd /home/wyh/projects/tetris-ai-benchmark-planning
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/home/wyh/projects/tetris-ai-benchmark/.venv/bin:/home/wyh/.nvm/versions/node/v24.18.0/bin:/home/wyh/.local/bin:/usr/local/bin:/usr/bin:/bin
mkdir -p runs/model_based_planning
exec 9>runs/model_based_planning/queue.lock
flock -n 9 || exit 0
timeout --signal=TERM --kill-after=60s 86400s python -u -m training.model_based_planning >>runs/model_based_planning/queue.log 2>&1
