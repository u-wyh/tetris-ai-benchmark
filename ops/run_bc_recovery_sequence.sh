#!/usr/bin/env bash
set -euo pipefail
cd /home/wyh/projects/tetris-ai-benchmark-sequence-recovery
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/home/wyh/projects/tetris-ai-benchmark/.venv/bin:/home/wyh/.nvm/versions/node/v24.18.0/bin:/home/wyh/.local/bin:/usr/local/bin:/usr/bin:/bin
mkdir -p runs/bc_recovery_sequence
exec 9>runs/bc_recovery_sequence/queue.lock
flock -n 9 || exit 0
timeout --signal=TERM --kill-after=60s 86400s python -u -m training.bc_recovery_sequence >>runs/bc_recovery_sequence/queue.log 2>&1
