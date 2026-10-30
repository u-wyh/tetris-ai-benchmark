#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
validation_host=${1:-100.91.227.47}
validation_run=runs/model_based_planning_web
exec 8>"$validation_run/validation.lock"
flock -n 8
validation_pid=$(.venv/bin/python -c 'import json;print(json.load(open("runs/model_based_planning_web/deployment/api-handoff.json"))["new_pid"])')
validation_failed() {
  .venv/bin/python - <<'PYFAIL'
import json,datetime
from pathlib import Path
now=datetime.datetime.now().astimezone().isoformat()
Path('runs/model_based_planning_web/validation-failure.json').write_text(json.dumps({'at':now,'error':'validation command failed; see stage logs'}))
with open('OPERATIONS.md','a') as f:f.write(now+' | Public Lookahead web: background validation stopped on error; services retained. Evidence:runs/model_based_planning_web/validation-failure.json and stage logs.\n')
PYFAIL
}
trap validation_failed ERR
timeout 20m node tests/integration/lookahead-browser.cjs 300 3102841000 "http://$validation_host:8080" "$validation_run/production300" "$validation_pid" '' frozen > "$validation_run/production300.log" 2>&1
.venv/bin/python - <<'PY'
import json
r=json.load(open('runs/model_based_planning_web/production300/result.json'))
assert r['pieces']==r['core_checked_steps']==r['frozen_checked_steps']==300 and not r['errors']
PY
timeout 60m node tests/integration/lookahead-browser.cjs 10000 3102841001 "http://$validation_host:8080" "$validation_run/production10000" "$validation_pid" > "$validation_run/production10000.log" 2>&1
timeout 10m node tests/integration/lookahead-controls.cjs "http://$validation_host:8080" "$validation_run/controls.json" > "$validation_run/controls.log" 2>&1
.venv/bin/python - <<'PY'
import datetime,json,shutil
from pathlib import Path
from training.evaluation.web_longrun_verify import atomic
now=datetime.datetime.now().astimezone().isoformat()
failure=Path('runs/model_based_planning_web/validation-failure.json')
if failure.exists():
    archive=Path('runs/model_based_planning_web/failures/previous-validation-failure.json')
    archive.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(failure,archive)
    failure.unlink()
atomic(Path('runs/model_based_planning_web/validation-closed.json'),json.dumps({'at':now,'result':'completed'}).encode())
with open('OPERATIONS.md','a') as f:f.write(now+' | Public Lookahead web: production Chromium300 frozen-action/Core check, capped10K stability run and UI/fault/concurrent-browser suite completed. Evidence:runs/model_based_planning_web/production300/result.json,production10000/result.json,controls.json.\n')
PY
