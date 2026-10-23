# Fast PPO branch replay at observed game over

Completed 2026-10-23 13:01 +0800. The frozen [v2 protocol](../../runs/ppo_fast_branch_terminal_20261023/protocol.json) checked two existing PPO Top2 snapshots whose archived branches ended in game over after 500 pieces, before the archived 10,000-piece horizon. Snapshot, archive, and frozen PPO model SHA256 values were verified before replay. These are development cases, not new confirmation seeds.

| Seed | Action | Legacy | Fast | Outcome |
|---:|---:|---:|---:|---|
| 1541006 | 1003 | 1.691 s | 1.083 s | 500 pieces, game over |
| 1540012 | 1169 | 1.696 s | 1.076 s | 500 pieces, game over |

For both cases, complete legacy, fast, and archived outcome dictionaries matched exactly, including score and four-clear series. The two-run times are diagnostic only; the completed 1.433× paired throughput benchmark remains the performance estimate. Raw identities and outputs: `runs/ppo_fast_branch_terminal_20261023/`.

The initial v1 protocol mistakenly set the replay horizon to the observed death length, 500, and stopped at an identity assertion before replay. The corrected v2 protocol preserves v1 and uses the archived 10,000-piece horizon. No training, formal Core, model, or web-default changes.
