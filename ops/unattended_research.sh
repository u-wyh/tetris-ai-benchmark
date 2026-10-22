#!/usr/bin/env bash
set -euo pipefail

repo=/home/wyh/projects/tetris-ai-benchmark-branch-perf
state_dir=/home/wyh/projects/tetris-ai-benchmark/runs/unattended_research
interval_seconds=18000
mkdir -p "$state_dir"

exec 9>"$state_dir/agent.lock"
flock -n 9 || exit 0
if [[ -f "$state_dir/quota_exhausted" ]]; then
  exit 0
fi
now_epoch=$(date +%s)
if [[ -f "$state_dir/last_attempt_epoch" ]]; then
  last_epoch=$(<"$state_dir/last_attempt_epoch")
  if [[ "$last_epoch" =~ ^[0-9]+$ ]] && (( now_epoch - last_epoch < interval_seconds )); then
    exit 0
  fi
fi
printf '%s\n' "$now_epoch" > "$state_dir/last_attempt_epoch.tmp"
mv "$state_dir/last_attempt_epoch.tmp" "$state_dir/last_attempt_epoch"
stamp=$(date +%Y%m%dT%H%M%S%z)
log="$state_dir/$stamp.log"
{
  printf 'started=%s host=%s branch=%s\n' "$(date -Is)" "$(hostname)" "$(git -C "$repo" branch --show-current)"
  set +e
  timeout --signal=TERM --kill-after=60s 10800s \
    /home/wyh/.local/bin/codex exec --approve-for-me \
    -C "$repo" - < "$repo/ops/unattended_research_prompt.md"
  result=$?
  set -e
  printf 'finished=%s exit=%s\n' "$(date -Is)" "$result"
} >> "$log" 2>&1
if grep -Eiq 'insufficient_quota|quota exceeded|usage limit reached|hit your usage limit|out of credits|not enough credits' "$log"; then
  printf 'detected=%s log=%s\n' "$(date -Is)" "$log" > "$state_dir/quota_exhausted.tmp"
  mv "$state_dir/quota_exhausted.tmp" "$state_dir/quota_exhausted"
fi
exit "$result"
