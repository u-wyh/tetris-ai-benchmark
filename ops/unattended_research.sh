#!/usr/bin/env bash
set -euo pipefail

repo=/home/wyh/projects/tetris-ai-benchmark-branch-perf
state_dir=/home/wyh/projects/tetris-ai-benchmark/runs/unattended_research
deadline=2026-10-28T00:00:00+08:00
mkdir -p "$state_dir"

# Cron is kept installed so the audit trail remains visible after the five days.
if (( $(date +%s) >= $(date -d "$deadline" +%s) )); then
  exit 0
fi

exec 9>"$state_dir/agent.lock"
flock -n 9 || exit 0
stamp=$(date +%Y%m%dT%H%M%S%z)
log="$state_dir/$stamp.log"
{
  printf 'started=%s host=%s branch=%s\n' "$(date -Is)" "$(hostname)" "$(git -C "$repo" branch --show-current)"
  set +e
  timeout --signal=TERM --kill-after=60s 10800s \
    /home/wyh/.local/bin/codex exec --approve-for-me -s workspace-write \
    -C "$repo" - < "$repo/ops/unattended_research_prompt.md"
  result=$?
  set -e
  printf 'finished=%s exit=%s\n' "$(date -Is)" "$result"
  exit "$result"
} >> "$log" 2>&1
