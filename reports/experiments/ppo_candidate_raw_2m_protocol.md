# Candidate-Scoring PPO Raw 2M protocol

The candidate experiment uses the existing 10×20 TetrisCore, 7-Bag, SRS, Hold,
Top Out, 1840 Action IDs, legal mask, Raw placement Reward, and validation seed
file. Only the observation encoding and policy network change. It starts from
seed 42 with independent random weights; Raw MLP weights cannot be reused.

Observation version: `candidate-dict-state237-features16-v1`. Candidate version:
`candidate-placement-uint8-v1`. Every candidate feature is an exact bounded
integer in uint8; illegal rows contain zero and cannot enter the masked
categorical distribution. The 16 feature names and bounds are frozen in
`training/candidate_features.py`. They come from the public board, active/Hold
piece, visible Next when Hold is empty, and the deterministic current placement.
No hidden queue, bag, RNG, or future spawn participates in feature generation.

Formal run: `runs/ppo_candidate_raw_2m_seed42`, CUDA, eight workers, seed
42–49, 512 samples per worker per Task, 2,002,944 committed steps, original
Raw PPO learning rate/epochs/batch size/discount/entropy settings. The
transaction trainer saves model, optimizer, global RNG, worker Core states,
observation/mask digests, Task metrics, last three ordinary checkpoints,
milestones, baseline and best selection. A separate
`runs/ppo_candidate_pipeline_seed42/status.json` advances through feature
tests, performance smoke, 2M training, paired evaluation, report, and
completion. A failed stage is saved and can be retried without repeating
committed Tasks.

The 8-worker CUDA smoke performs two 4096-step Tasks with an intentional
checkpoint-boundary restart and compares its final parameters, optimizer,
RNG and worker digests with an uninterrupted two-Task control. It records
training steps/s and PyTorch peak allocated/reserved GPU memory. The full
2M comparison uses the same 32 validation seeds `100000..100031`, a 5000-piece
cap, deterministic inference, and official masks for both Candidate and Raw.
The final-test seeds `200000..200099` are excluded. The pipeline does not
start a 10M run.
