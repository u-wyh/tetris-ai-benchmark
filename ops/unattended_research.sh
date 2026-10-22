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
probe_log="$state_dir/$stamp.probe.log"
probe_reply="$state_dir/$stamp.probe.reply"

# A successful login only checks stored credentials. This small model call checks
# whether the account can actually run before starting the research milestone.
set +e
timeout --signal=TERM --kill-after=10s 120s \
  /home/wyh/.local/bin/codex exec --ephemeral --approve-for-me \
  -C "$repo" -o "$probe_reply" '请只回复 READY。' > "$probe_log" 2>&1
probe_result=$?
set -e
if grep -Eiq 'insufficient_quota|quota exceeded|usage limit reached|hit your usage limit|out of credits|not enough credits' "$probe_log"; then
  printf 'detected=%s log=%s\n' "$(date -Is)" "$probe_log" > "$state_dir/quota_exhausted.tmp"
  mv "$state_dir/quota_exhausted.tmp" "$state_dir/quota_exhausted"
  exit 0
fi
if (( probe_result != 0 )) || [[ ! -f "$probe_reply" ]] || ! grep -Fxq READY "$probe_reply"; then
  printf 'probe_failed=%s exit=%s log=%s\n' "$(date -Is)" "$probe_result" "$probe_log" > "$state_dir/$stamp.status"
  exit 0
fi
{
  printf 'started=%s host=%s branch=%s probe=%s\n' "$(date -Is)" "$(hostname)" "$(git -C "$repo" branch --show-current)" "$probe_log"
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
