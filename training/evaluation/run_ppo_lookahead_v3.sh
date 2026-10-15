#!/usr/bin/env bash
set -euo pipefail
cd /home/wyh/projects/tetris-ai-benchmark
mkdir -p runs/ppo30m_lookahead_v3
.venv/bin/python -m pytest -q > runs/ppo30m_lookahead_v3/python_tests.log 2>&1
.venv/bin/python -m unittest discover -s tests/parity -v > runs/ppo30m_lookahead_v3/parity_tests.log 2>&1
node --test tests/*.test.cjs > runs/ppo30m_lookahead_v3/js_tests.log 2>&1
.venv/bin/python -m training.evaluation.ppo_lookahead_v3_queue > runs/ppo30m_lookahead_v3/queue.log 2>&1
