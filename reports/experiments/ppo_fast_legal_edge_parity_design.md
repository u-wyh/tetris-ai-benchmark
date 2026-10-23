# Evaluation legal placement edge parity protocol

Frozen 2026-10-23 12:54 +0800, before running the new test. Compare the
evaluation-only bitboard BFS against the formal Core placement generator using
exact ordered placement dictionaries, including paths and occupied cells.

Cases: seven current pieces; four fixed board patterns (empty, low staggered,
deep well, and near-top irregular); hold unavailable, hold with an existing
piece, and hold with an empty slot. This yields 84 public states. Existing
seeded trajectory tests remain separate. The milestone passes only if all
states match exactly and the existing fast environment tests pass. No training,
model changes, or formal Core changes are authorized by this protocol.
