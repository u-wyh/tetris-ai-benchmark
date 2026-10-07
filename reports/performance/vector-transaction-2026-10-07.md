# CUDA 8-worker transactional PPO recovery

New run: `ppo_transaction_vec8_cuda_test_seed42`; original single-worker run remains untouched. Config: CUDA, `forkserver` SubprocVecEnv, 8 workers seeded 42–49, `n_steps=512`, one 4096-sample rollout/update per Task, 32768 target steps. Tetris rules, Observation, Action Space and reward were not changed.

Each Task commits `model.zip` (including optimizer), main CPU/CUDA RNG, eight `TetrisEnv` logical states, metrics, TensorBoard files, per-worker observation/mask digests and a SHA256 file manifest. The commit pointer advances after fsynced Task files are atomically renamed. Resume verifies the pointer's Task, abandons uncommitted/orphaned work, restores each worker by index and checks all digests before training.

Automated CUDA tests covered a partial Task 2, a fully written but uncommitted/orphaned Task 2, and a corrupt committed Task. Interrupted and uninterrupted 3-Task runs produced byte-identical eight-worker state snapshots and identical policy parameters, optimizer, timesteps, Python/NumPy/PyTorch CPU/CUDA RNG. All worker PIDs exited after normal completion, interruption and restore.

Two uninterrupted 3-Task comparisons measured steady Task 2–3 training at about 546–554 steps/s, 2–4% below the previous 566.83 steps/s nontransactional CUDA-8 benchmark. Recent Task checkpoint writes took 0.057–0.060 s; full commit including pointer/metrics took 0.076–0.085 s. Each committed Task occupied about 8.79 MB. No large steady-state slowdown or checkpoint growth was observed.

The real 32768-step background run is started only after code tests, commit and push; the user performs the reboot manually once at least one Task is committed and another is in progress. No systemd automation is configured.
