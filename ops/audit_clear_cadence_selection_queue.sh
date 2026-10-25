#!/usr/bin/env bash
set -euo pipefail
cd /home/wyh/projects/tetris-ai-benchmark-branch-perf
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run=runs/ppo_clear_cadence_counterfactual/selection_audit
mkdir -p "$run"
for seed in {1550000..1550007}; do
  if ! taskset -c 15 /home/wyh/projects/tetris-ai-benchmark/.venv/bin/python \
      -m training.evaluation.audit_clear_cadence_selection "$seed" \
      > "$run/seed_${seed}.log" 2>&1; then
    printf '%s | 清行节奏采样规则重放审计失败：seed %s；停止队列。证据：%s/seed_%s.log。\n' \
      "$(date -Is)" "$seed" "$run" "$seed" >> OPERATIONS.md
    exit 1
  fi
  printf '%s | 清行节奏采样规则重放审计通过：seed %s；核对全部检查点、入选与排除计数及终局。证据：%s/seed_%s.json。\n' \
    "$(date -Is)" "$seed" "$run" "$seed" >> OPERATIONS.md
done
