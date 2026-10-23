# Evaluation candidate feature edge parity protocol

Frozen 2026-10-23 12:56 +0800 before running this validation. Compare the evaluation-only fast candidate feature implementation with the formal `training.candidate_features.candidate_features` reference, using the same formal Core legal placements. Require exact equality of the full 1840-action feature matrix and mask for each case.

Cases: the existing seven pieces × four fixed edge boards × three hold states (84 public states), plus three existing archived PPO Top2 selected-state snapshots with seed labels 1540000, 1540004, and 1541002. The snapshots have SHA256 values `51b53f8b5c38be0cc9bb1901c3a9ceb3030edb638c1ebe86cccd9e3bd82a93af`, `3f17b8430847dfc66c0fa248f22f383f7b091f13fa25c53d99478f8ae6143148`, and `7617f701a243268dc1edcc6e51f1767fbd6ffb4e16f47a310860440c18a26660`, respectively. These are previously used development snapshots, not fresh confirmation seeds.

Pass criterion: all 87 cases agree exactly and the existing targeted fast candidate tests pass. Record any empty-placement cases. This is a correctness validation, with no speed or model-quality claim. No new training, formal Core, model, or web-default changes.
