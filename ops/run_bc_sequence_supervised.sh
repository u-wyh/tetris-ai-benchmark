#!/usr/bin/env bash
set -euo pipefail
cd /home/wyh/projects/tetris-ai-benchmark-sequence
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/home/wyh/projects/tetris-ai-benchmark/.venv/bin:/home/wyh/.nvm/versions/node/v24.18.0/bin:/home/wyh/.local/bin:/usr/local/bin:/usr/bin:/bin
mkdir -p runs/bc_sequence_transfer
exec 9>runs/bc_sequence_transfer/queue.lock
flock -n 9 || exit 0
exec timeout --signal=TERM --kill-after=30s 43200s python -u -m training.bc_sequence_supervised >> runs/bc_sequence_transfer/supervised_queue.log 2>&1
