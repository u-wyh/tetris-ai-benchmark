#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH="$PWD/.venv/bin:/home/wyh/.nvm/versions/node/v24.18.0/bin:/usr/local/bin:/usr/bin:/bin"
mkdir -p runs/lookahead_long_validation
exec 9>runs/lookahead_long_validation/queue.lock
flock -n 9 || exit 0
timeout 600s python -m pytest -q tests/test_lookahead_long_validation.py tests/test_lookahead_ablation.py >runs/lookahead_long_validation/preflight_tests.log 2>&1
for long_validation_attempt in 0 1; do
  long_validation_remaining=$(python -c 'import json,time;print(max(1,int(129600-time.time()+json.load(open("runs/lookahead_long_validation/task_started.json"))["epoch"])))')
  set +e
  timeout --signal=TERM --kill-after=60s "${long_validation_remaining}s" python -u -m training.lookahead_long_validation >>runs/lookahead_long_validation/queue.log 2>&1
  long_validation_code=$?
  set -e
  if test -f runs/lookahead_long_validation/closed.json; then exit "$long_validation_code"; fi
  if test "$long_validation_code" -eq 0; then exit 0; fi
  if test "$long_validation_attempt" -eq 0 && test "$long_validation_remaining" -gt 300 && ! test -f runs/lookahead_long_validation/stop.json; then
    python - <<'PY'
from datetime import datetime
with open('OPERATIONS.md','a') as f:f.write(datetime.now().astimezone().isoformat()+' | Lookahead long validation: abnormal process exit without closed/stop marker; one bounded atomic-checkpoint resume, same source/config/seed budget. Evidence:runs/lookahead_long_validation/queue.log.\n')
PY
  else exit "$long_validation_code"; fi
done
