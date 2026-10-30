#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="$PWD/.venv/bin:/home/wyh/.nvm/versions/node/v24.18.0/bin:/usr/local/bin:/usr/bin:/bin"
mkdir -p runs/lookahead_ablation
exec 9>runs/lookahead_ablation/queue.lock
flock -n 9 || exit 0
python -m pytest -q tests/test_lookahead_ablation.py >runs/lookahead_ablation/preflight_tests.log 2>&1
for ablation_attempt in 0 1; do
  ablation_remaining=$(python -c 'import json,time;print(max(1,int(86400-time.time()+json.load(open("runs/lookahead_ablation/task_started.json"))["epoch"])))')
  set +e
  timeout --signal=TERM --kill-after=60s "${ablation_remaining}s" python -u -m training.lookahead_ablation >>runs/lookahead_ablation/queue.log 2>&1
  ablation_code=$?
  set -e
  if test -f runs/lookahead_ablation/closed.json; then exit "$ablation_code"; fi
  if test "$ablation_code" -eq 0; then exit 0; fi
  if test "$ablation_attempt" -eq 0 && test "$ablation_remaining" -gt 300 && ! test -f runs/lookahead_ablation/stop.json; then
    python - <<'PY'
from datetime import datetime
with open('OPERATIONS.md','a') as f:f.write(datetime.now().astimezone().isoformat()+' | Lookahead ablation: abnormal process exit without closed/stop marker; one bounded atomic-checkpoint resume, same source/config/seed budget. Evidence:runs/lookahead_ablation/queue.log.\n')
PY
  else exit "$ablation_code"; fi
done
