"""Vectorized clearing and features match independent cell geometry and trajectories."""
import pickle
import os
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from training.candidate_features import board_features
from training.evaluation.fast_candidate_features import candidate_features as reference
from training.native_candidate_features import NativeCandidateFeatures
from training.train_vector_transaction import TransactionCompactSearchCandidateEnv
from training.tetris_core import TetrisCore
from training.tetris_core.pieces import make_piece


@pytest.fixture(scope='module', autouse=True)
def native_runtime():
    global batched
    manifest = Path(os.environ['TETRIS_NATIVE_MANIFEST'])
    batched = NativeCandidateFeatures(manifest, expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    return batched


class NativeTestEnv(TransactionCompactSearchCandidateEnv):
    def _features_and_mask(self):
        if self.core is None:
            raise RuntimeError('Call reset first')
        if self._candidate_cache is None:
            public = self.core.get_public_observation()
            self._placement_cache = self.placement_search(public)
            self._candidate_cache = batched(public, self._placement_cache)
        return self._candidate_cache


def test_canonical_clearing_and_invisible_candidates_do_not_use_reference_fallback(monkeypatch):
    import training.native_candidate_features as implementation
    core = TetrisCore(42)
    for y in range(16, 20):
        core.board[y] = ['Z' if x != 4 else None for x in range(10)]
    placement = dict(actionId=0, hold=0, rotation=1, x=3, y=16,
                     occupiedCells=[dict(x=4, y=y) for y in range(16, 20)])
    public = core.get_public_observation()
    expected = reference(public, [placement])
    monkeypatch.setattr(implementation, 'reference', lambda *args: pytest.fail('Unexpected reference fallback'))
    actual = batched(public, [placement])
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
    core.board[19] = ['Z']*10
    invisible = dict(placement, y=-1, occupiedCells=[dict(x=x, y=-1) for x in range(4)])
    expected = reference(core.get_public_observation(), [invisible])
    actual = batched(core.get_public_observation(), [invisible])
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)


def equal(public, placements):
    expected, actual = reference(public, placements), batched(public, placements)
    for left, right in zip(expected, actual):
        assert left.dtype == right.dtype and left.shape == right.shape
        np.testing.assert_array_equal(left, right)


def test_random_columns_and_line_compaction_against_independent_cell_reference():
    from training.candidate_features import candidate_features as cell_reference
    rng = np.random.default_rng(316013)
    core = TetrisCore(42)
    placement = dict(actionId=0, hold=0, rotation=0, x=3, y=0,
                     occupiedCells=[dict(x=3, y=0)]*4)
    columns = rng.integers(0, 1 << 20, size=(1024,10), dtype=np.uint32)
    boundaries = [0, (1 << 20)-1]+[1 << k for k in range(20)]+[(1 << k)-1 for k in range(21)]
    boards = [[['Z' if int(row[x]) & (1 << (19-y)) else None for x in range(10)] for y in range(20)]
              for row in list(columns)+[[value]*10 for value in boundaries]]
    for index in range(128):
        board = [['Z' if rng.integers(2) else None for x in range(10)] for y in range(20)]
        for y in rng.choice(20, size=index % 21, replace=False): board[y] = ['Z']*10
        boards.append(board)
    for index, board in enumerate(boards):
        core.board = board
        public = core.get_public_observation()
        p = dict(placement, occupiedCells=[dict(x=3,y=-1 if index % 3 == 0 else 0)]*4)
        try:
            expected = cell_reference(public, [p])
        except OverflowError:
            with pytest.raises(OverflowError): batched(public, [p])
        else:
            for left,right in zip(expected,batched(public,[p])):
                np.testing.assert_array_equal(left,right)


@pytest.mark.parametrize('kind', 'IJLOSTZ')
@pytest.mark.parametrize('pattern', range(4))
@pytest.mark.parametrize('hold', (None, 'T', 'I'))
def test_fixed_boards_holds_and_piece_types(kind, pattern, hold):
    core = TetrisCore(42)
    core.current, core.hold = make_piece(kind), hold
    if pattern:
        for y in range((pattern-1)*4, 20):
            for x in range(10):
                if (3*x+7*y) % (pattern*3+2) < pattern:
                    core.board[y][x] = 'Z'
    equal(core.get_public_observation(), core.get_legal_placements())


@pytest.mark.parametrize('rows', [(19,), (18, 19), (16, 17, 18, 19), (16, 18, 19)])
def test_single_multiple_and_nonconsecutive_line_clears(rows):
    core = TetrisCore(42)
    for y in rows:
        for x in range(10):
            if x != 4:
                core.board[y][x] = 'Z'
    placement = dict(actionId=0, hold=0, rotation=1, x=3, y=16,
        occupiedCells=[dict(x=4, y=y) for y in (16, 17, 18, 19)])
    equal(core.get_public_observation(), [placement])


@pytest.mark.parametrize('cells', [[], [dict(x=x, y=-1) for x in range(4)],
                                 [dict(x=3, y=-1), dict(x=3, y=0)]])
def test_existing_full_rows_and_fully_invisible_placement(cells):
    core = TetrisCore(42)
    core.board[19] = ['Z']*10
    placement = dict(actionId=0, hold=0, rotation=0, x=3, y=-1, occupiedCells=cells)
    equal(core.get_public_observation(), [placement])


@pytest.mark.parametrize('fixture', ['single_clear', 'lock_out', 'partial_lock'])
def test_actual_topout_and_clear_edges(fixture):
    core = TetrisCore(42)
    core.current = make_piece('O' if fixture != 'lock_out' else 'I')
    if fixture == 'single_clear':
        for x in range(10):
            if x not in (3, 4):
                core.board[19][x] = 'Z'
    elif fixture == 'lock_out':
        core.current['y'] = -2
        for x in range(3, 7):
            core.board[0][x] = 'Z'
    else:
        core.current.update(x=0, y=-1)
        core.board[1][0] = 'Z'
    equal(core.get_public_observation(), core.get_legal_placements())


@pytest.mark.parametrize('patch', [dict(x=-1), dict(x=10), dict(y=-4), dict(y=20),
                                   dict(rotation=-1), dict(rotation=4), dict(hold=2)])
def test_invalid_feature_ranges(patch):
    core = TetrisCore(42)
    placement = dict(core.get_legal_placements()[0], **patch)
    for method in (reference, batched):
        with pytest.raises(OverflowError):
            method(core.get_public_observation(), [placement])


def test_duplicate_action_rejected():
    core = TetrisCore(42)
    placement = core.get_legal_placements()[0]
    for method in (reference, batched):
        with pytest.raises(AssertionError, match='Duplicate'):
            method(core.get_public_observation(), [placement, placement])


def test_empty_variable_cell_counts_and_negative_action_indices_match_reference():
    core = TetrisCore(42)
    public = core.get_public_observation()
    placement = core.get_legal_placements()[0]
    equal(public, [])
    equal(public, [dict(placement, actionId=-1)])
    for count in range(6):
        cells = [dict(x=4, y=18)]*count
        equal(public, [dict(placement, occupiedCells=cells)])


@pytest.mark.parametrize('patch', [dict(actionId=1840), dict(actionId=-1841),
    dict(actionId=10**30), dict(x=10**30), dict(rotation=1.5),
    dict(occupiedCells=[dict(x=-1, y=18)]*4), dict(occupiedCells=[dict(x=4, y=20)]*4)])
def test_exception_type_matches_reference_or_fractional_fallback_matches_values(patch):
    core = TetrisCore(42)
    public = core.get_public_observation()
    placement = dict(core.get_legal_placements()[0], **patch)
    try:
        expected = reference(public, [placement])
    except Exception as error:
        with pytest.raises(type(error)):
            batched(public, [placement])
    else:
        actual = batched(public, [placement])
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)


def test_trajectory_masks_rewards_and_snapshot_restore_exact():
    left, right = TransactionCompactSearchCandidateEnv(max_pieces=50), NativeTestEnv(max_pieces=50)
    rng = np.random.default_rng(286014)
    try:
        observations = [env.reset(seed=286015)[0] for env in (left, right)]
        for _ in range(200):
            for name in observations[0]:
                np.testing.assert_array_equal(observations[0][name], observations[1][name])
            np.testing.assert_array_equal(left.action_masks(), right.action_masks())
            if left.pieces == 20:
                right.set_state(pickle.loads(pickle.dumps(left.get_state(), protocol=5)))
            action = int(rng.choice(np.flatnonzero(left.action_masks())))
            result = [env.step(action) for env in (left, right)]
            assert result[0][1:] == result[1][1:]
            observations = [r[0] for r in result]
            if result[0][2] or result[0][3]:
                observations = [env.reset()[0] for env in (left, right)]
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize('mutation', ['manifest', 'binary', 'source', 'wrapper', 'abi', 'flags', 'missing'])
def test_runtime_mismatches_fail_closed(tmp_path, mutation):
    original = Path(os.environ['TETRIS_NATIVE_MANIFEST'])
    manifest = json.loads(original.read_text())
    binary = tmp_path/'candidate_features.so'
    binary.write_bytes(Path(manifest['binary']).read_bytes())
    manifest['binary'] = str(binary)
    if mutation == 'binary': binary.write_bytes(b'corrupted')
    if mutation == 'missing': binary.unlink()
    if mutation in ('source', 'wrapper'): manifest[mutation+'_sha256'] = '0'*64
    if mutation == 'abi': manifest['abi'] = 'wrong'
    if mutation == 'flags': manifest['flags'] = ['-O0']
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    expected = '0'*64 if mutation == 'manifest' else hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((ValueError, OSError)):
        NativeCandidateFeatures(path, expected_manifest_sha256=expected)


@pytest.mark.parametrize('patch', [dict(actionId=np.int64(-1840)), dict(actionId=True),
    dict(hold=0.5), dict(y=-2**63), dict(y=2**63-1), dict(x=2**63),
    dict(occupiedCells=[dict(x=2**63-1,y=19)]*4),
    dict(occupiedCells=[dict(x=0,y=2**63-1)]*4),
    dict(occupiedCells=[dict(x=2**63-1,y=-2**63)]*4)])
def test_signed_limits_and_unusual_inputs_preserve_reference(patch):
    core = TetrisCore(42)
    public = core.get_public_observation()
    placement = dict(core.get_legal_placements()[0], **patch)
    try:
        expected = reference(public,[placement])
    except Exception as error:
        with pytest.raises(type(error)): batched(public,[placement])
    else:
        for left,right in zip(expected,batched(public,[placement])):
            np.testing.assert_array_equal(left,right)
