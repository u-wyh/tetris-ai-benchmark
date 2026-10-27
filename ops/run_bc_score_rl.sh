#!/usr/bin/env bash
set -euo pipefail
cd /home/wyh/projects/tetris-ai-benchmark-bc-score
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/home/wyh/projects/tetris-ai-benchmark/.venv/bin:/home/wyh/.nvm/versions/node/v24.18.0/bin:/home/wyh/.local/bin:/usr/local/bin:/usr/bin:/bin
export HTTP_PROXY=http://127.0.0.1:10809 HTTPS_PROXY=http://127.0.0.1:10809
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" NO_PROXY=localhost,127.0.0.1,::1
mkdir -p runs/bc_score_rl
exec 9>runs/bc_score_rl/queue.lock
flock -n 9 || exit 0
for attempt in 1 2 3; do
  set +e
  timeout --signal=TERM --kill-after=60s 172800s python -m training.bc_score_rl >> runs/bc_score_rl/queue.log 2>&1
  result=$?
  set -e
  printf '%s supervisor attempt=%s exit=%s\n' "$(date -Is)" "$attempt" "$result" >> runs/bc_score_rl/queue.log
  case "$result" in
    75|130|137|143) sleep 10 ;;
    *) exit "$result" ;;
  esac
done
exit 1
