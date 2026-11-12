# Traditional AI long-life experiment protocol

This experiment uses the unchanged Python TetrisCore and the frozen V1 Adapter / V2 Hold-only / V2 Beam-8 strategies. The first pilot uses validation seeds 100016–100018 at a 10,000-piece cap. The formal run uses 100016–100031 at the same cap. Any higher-cap extension must rerun all three policies on the same seed subset and cap. Final-test seeds are excluded.

Each policy's existing CLI atomically writes a completed `seed_<id>.json`, `aggregate.json`, `report.md`, `config.json`, and `status.json` in its own new directory. A partial episode appears only in `status.json` and is replayed from the seed after interruption. `--resume` requires identical code/configuration and does not count partial episodes. Logs live beside each output directory. No PPO process or checkpoint is touched.

From the repository root, start one low-priority tmux session per mode. Replace `<run>` with a new parent directory and `<seeds>` with a space-separated seed list. Create only the parent directory; the benchmark creates each mode subdirectory.

```bash
mkdir -p reports/traditional_longlife/<run>
tmux new-session -d -s longlife-v1 -c "$PWD" 'nice -n 10 .venv/bin/python -m training.evaluation.traditional --output reports/traditional_longlife/<run>/v1 --max-pieces 10000 --seeds <seeds> > reports/traditional_longlife/<run>/v1.log 2>&1'
tmux new-session -d -s longlife-hold -c "$PWD" 'nice -n 10 .venv/bin/python -m training.evaluation.traditional_v2 --mode hold --output reports/traditional_longlife/<run>/hold --max-pieces 10000 --seeds <seeds> > reports/traditional_longlife/<run>/hold.log 2>&1'
tmux new-session -d -s longlife-beam -c "$PWD" 'nice -n 10 .venv/bin/python -m training.evaluation.traditional_v2 --mode beam --output reports/traditional_longlife/<run>/beam --max-pieces 10000 --seeds <seeds> > reports/traditional_longlife/<run>/beam.log 2>&1'
```

Status and resume (use the same seeds and cap as the original run):

```bash
tmux ls
.venv/bin/python -m training.evaluation.traditional --output reports/traditional_longlife/<run>/v1 --status
.venv/bin/python -m training.evaluation.traditional_v2 --output reports/traditional_longlife/<run>/hold --status
.venv/bin/python -m training.evaluation.traditional_v2 --output reports/traditional_longlife/<run>/beam --status
# If a session stopped, restart its original command with --resume and a new tmux session.
```

After all three status files say `complete`, generate cap-aware paired results and public-state disagreement fixtures in a new output directory:

```bash
.venv/bin/python -m training.evaluation.longlife \
  --v1-dir reports/traditional_longlife/<run>/v1 \
  --hold-dir reports/traditional_longlife/<run>/hold \
  --beam-dir reports/traditional_longlife/<run>/beam \
  --output reports/traditional_longlife/<run>_analysis --disagreements --score-breakdown

.venv/bin/python -m training.evaluation.profile_longlife \
  --reference reports/traditional_longlife/<run>/hold/seed_100016.json \
  --output reports/traditional_longlife/<run>_matched_profile
```

The paired report compares restricted mean pieces `min(lifetime, cap)` and resamples whole seed pairs for its 95% interval. A pair in which both runs cap is unresolved for ultimate lifetime. Equal-piece score is reconstructed at the shorter observed episode length; it should not be confused with final score from unequal-length games. The disagreement fixture records only public observations for decision analysis. A separate Core fork supplies identical hidden initial state for short counterfactual follow-up, while both decision policies still receive only public observations.

To extend a completed same-cap pilot without rerunning its seeds, create each new mode directory with `training.evaluation.extend_traditional`. It validates the source and imports only complete seed JSON files, recording a manifest. Then start the original benchmark CLI with the full seed list and `--resume` in tmux. Example for V1 (repeat for Hold and Beam with `training.evaluation.traditional_v2 --mode hold/beam`):

```bash
.venv/bin/python -m training.evaluation.extend_traditional \
  --source reports/traditional_longlife/<pilot>/v1 \
  --output reports/traditional_longlife/<formal>/v1 \
  --seeds {100016..100031}
tmux new-session -d -s longlife-v1-formal -c "$PWD" 'nice -n 10 .venv/bin/python -m training.evaluation.traditional --output reports/traditional_longlife/<formal>/v1 --max-pieces 10000 --seeds {100016..100031} --resume > reports/traditional_longlife/<formal>/v1.log 2>&1'
```

For a slow formal mode, run disjoint seed shards in separate low-priority tmux sessions instead of the single resume command above. After every shard reports `complete`, import all shards into the prepared target while no target benchmark process is running, then resume once to write the combined aggregate/report. `merge_traditional` refuses duplicate seed files and records source directories and Git commits:

```bash
.venv/bin/python -m training.evaluation.merge_traditional \
  --target reports/traditional_longlife/<formal>/v1 \
  --sources reports/traditional_longlife/<formal>/v1_shard_a \
            reports/traditional_longlife/<formal>/v1_shard_b
.venv/bin/python -m training.evaluation.traditional \
  --output reports/traditional_longlife/<formal>/v1 \
  --max-pieces 10000 --seeds {100016..100031} --resume
```

## 数据入库说明（2026-11-12）

`reports/traditional_longlife/` 纳入 Git，但 **100K 块的逐 seed 轨迹不提交**：
`*_100000_*/{v1,hold,beam}/seed_*.json` 每个约 2.4 MB（含完整动作轨迹），
由 `.gitignore` 排除，仅保留本机副本。对外的聚合结果在各自目录的
`aggregate.json`、`config.json`、`report.md` 与分析目录里；重新生成需要重跑对应
的传统评测（每局 100K 块，耗时以 README 上文估算为准）。
