#!/usr/bin/env bash
set -euo pipefail

repo=/home/wyh/projects/tetris-ai-benchmark-branch-perf
state_dir=/home/wyh/projects/tetris-ai-benchmark/runs/unattended_research
session=tetris-unattended-research
quota_retry_seconds=18000
error_retry_seconds=900
mkdir -p "$state_dir"

# Cron does not load the interactive shell environment on this server.
export PATH=/home/wyh/projects/tetris-ai-benchmark/.venv/bin:/home/wyh/.nvm/versions/node/v24.18.0/bin:/home/wyh/.local/bin:/usr/local/bin:/usr/bin:/bin
export CODEX_HOME=/home/wyh/.codex
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export HTTP_PROXY=http://127.0.0.1:10809
export HTTPS_PROXY="$HTTP_PROXY"
export ALL_PROXY="$HTTP_PROXY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTP_PROXY"
export all_proxy="$HTTP_PROXY"
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="$NO_PROXY"

if [[ "${1:-}" != --run ]]; then
  # Cron is a watchdog. The long-running worker lives in tmux across SSH loss.
  if ! tmux has-session -t "$session" 2>/dev/null; then
    tmux new-session -d -s "$session" -c "$repo" "$repo/ops/unattended_research.sh --run"
  fi
  exit 0
fi

exec 9>"$state_dir/agent.lock"
flock -n 9 || exit 0

quota_message() {
  grep -Eiq 'insufficient_quota|quota exceeded|usage limit reached|hit your usage limit|out of credits|not enough credits|you have reached your .*limit' "$1"
}

need_probe=1
while true; do
  stamp=$(date +%Y%m%dT%H%M%S%z)
  probe_log="$state_dir/$stamp.probe.log"
  probe_reply="$state_dir/$stamp.probe.reply"
  log="$state_dir/$stamp.log"
  main_reply="$state_dir/$stamp.reply"

  if (( need_probe )); then
    # Login status alone does not show whether a model request can run.
    set +e
    timeout --signal=TERM --kill-after=10s 120s \
      /home/wyh/.local/bin/codex exec --ephemeral --approve-for-me \
      -C "$repo" -o "$probe_reply" '请只回复 READY。' > "$probe_log" 2>&1
    probe_result=$?
    set -e

    if quota_message "$probe_log"; then
      printf '%s quota_exhausted probe=%s retry_in_seconds=%s\n' "$(date -Is)" "$probe_log" "$quota_retry_seconds" >> "$state_dir/status.log"
      sleep "$quota_retry_seconds"
      continue
    fi
    if (( probe_result != 0 )) || [[ ! -f "$probe_reply" ]] || ! grep -Fxq READY "$probe_reply"; then
      printf '%s probe_failed exit=%s probe=%s retry_in_seconds=%s\n' "$(date -Is)" "$probe_result" "$probe_log" "$error_retry_seconds" >> "$state_dir/status.log"
      sleep "$error_retry_seconds"
      continue
    fi
    need_probe=0
  fi

  printf '%s research_started probe=%s\n' "$(date -Is)" "$probe_log" >> "$state_dir/status.log"
  set +e
  timeout --signal=TERM --kill-after=60s 10800s \
    /home/wyh/.local/bin/codex exec --approve-for-me \
    -C "$repo" -o "$main_reply" - < "$repo/ops/unattended_research_prompt.md" > "$log" 2>&1
  result=$?
  set -e
  printf '%s research_finished exit=%s log=%s\n' "$(date -Is)" "$result" "$log" >> "$state_dir/status.log"

  if quota_message "$log"; then
    printf '%s quota_exhausted main=%s retry_in_seconds=%s\n' "$(date -Is)" "$log" "$quota_retry_seconds" >> "$state_dir/status.log"
    need_probe=1
    sleep "$quota_retry_seconds"
  elif (( result != 0 )); then
    need_probe=1
    sleep "$error_retry_seconds"
  elif [[ -f "$main_reply" ]] && grep -Fxq NO_SAFE_WORK "$main_reply"; then
    printf '%s no_safe_work retry_in_seconds=%s\n' "$(date -Is)" "$quota_retry_seconds" >> "$state_dir/status.log"
    need_probe=1
    sleep "$quota_retry_seconds"
  else
    # A completed milestone immediately makes room for the next one.
    sleep 10
  fi
done
