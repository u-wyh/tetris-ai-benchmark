import copy
import json
from unittest.mock import patch

import pytest

from training.evaluation.traditional import aggregate, play, run
from training.evaluation.merge_traditional import merge
from training.tetris_core import TetrisCore
from training.tetris_core.pieces import make_piece
from training.traditional.v1 import V1Adapter, board_value, simulate
from training.traditional.legacy_heuristic import LegacyHeuristic, NAME


def test_formal_legacy_name():
    assert NAME == 'legacy_heuristic'
    assert LegacyHeuristic is V1Adapter


def test_formula_matches_browser():
    import subprocess
    core = TetrisCore(7)
    for x, height in enumerate([2, 5, 3, 0, 4, 6, 1, 2, 0, 3]):
        for y in range(20-height, 20):
            core.board[y][x] = 'T'
    core.board[18][1] = None
    script = ('const {game}=require("./tests/game-harness.cjs"); const g=game(); '
              f'console.log(g.run("boardValue("+JSON.stringify({json.dumps(core.board)})+",2)"));')
    expected = float(subprocess.check_output(['node', '-e', script], text=True))
    assert board_value(core.board, 2) == pytest.approx(expected, abs=1e-12)


def test_public_boundary_legal_no_hold_and_determinism():
    core = TetrisCore(123)
    public = core.get_public_observation()
    before = copy.deepcopy(public)
    public['next'] = public['next'][:1]  # policy needs only Next 1
    action = V1Adapter(123).choose(public)
    assert action == V1Adapter(123).choose(public)
    assert core.get_action_mask()[action] and action < 920
    assert core.get_public_observation() == before
    core.bag.reverse()
    core.queue[3:] = ['I', 'I']
    core.rng.state = 0
    assert action == V1Adapter(123).choose(core.get_public_observation())


def test_hold_only_rejected_and_above_board_fallback():
    public = TetrisCore(1).get_public_observation()
    with pytest.raises(ValueError, match='non-Hold'):
        V1Adapter(1).choose(public, [dict(hold=1, actionId=1000)])
    placement = dict(hold=0, actionId=3, occupiedCells=[dict(x=0, y=-1)])
    assert simulate(public['board'], placement) is None
    assert V1Adapter(1).choose(public, [placement]) == 3


def test_top_out_and_cap_termination():
    core = TetrisCore(1)
    core.board = [['T'] * 10 for _ in range(20)]
    core.current = make_piece('I')
    core.current['y'] = -2
    observation = core.get_public_observation()
    placements = core.get_legal_placements()
    action = V1Adapter(1).choose(observation, placements)
    core.step(action)
    assert core.phase == 'over'
    first, second = play(12, 3), play(12, 3)
    assert first['actions'] == second['actions']
    assert first['score'] == second['score']
    assert first['truncated'] and not first['game_over']
    assert len(first['decision_ms']) == 3
    metrics = aggregate([first, second])
    assert metrics['cap_rate'] == 1 and metrics['evaluation_saturated']


def test_resume_no_overwrite_and_exception_record(tmp_path):
    output = tmp_path / 'run'
    original_play = play
    def fail_second(seed, cap, progress):
        if seed == 2:
            raise RuntimeError('injected failure')
        return original_play(seed, cap, progress)
    with patch('training.evaluation.traditional.play', side_effect=fail_second):
        with pytest.raises(RuntimeError, match='injected'):
            run(output, [1, 2], 1)
    saved = (output / 'seed_1.json').read_bytes()
    assert not (output / 'seed_2.json').exists()
    assert list(output.glob('error_*.json'))
    with patch('training.evaluation.traditional.play', wraps=original_play) as mocked:
        run(output, [1, 2], 1, resume=True)
        assert mocked.call_count == 1
    assert (output / 'seed_1.json').read_bytes() == saved
    assert json.loads((output / 'status.json').read_text())['state'] == 'complete'
    with pytest.raises(FileExistsError):
        run(output, [1, 2], 1)
    with pytest.raises(ValueError, match='mismatch'):
        run(output, [1, 2], 2, resume=True)


def test_input_errors(tmp_path):
    for seeds, cap in [([1, 1], 1), ([-1], 1), ([1], 0), ([], 1)]:
        with pytest.raises(ValueError):
            run(tmp_path / 'invalid', seeds, cap)


def test_merge_compatible_completed_seed_only(tmp_path):
    target, source = tmp_path / 'target', tmp_path / 'source'
    run(target, [1, 2], 1)
    run(source, [2], 1)
    (target / 'seed_2.json').unlink()
    (target / 'aggregate.json').unlink()
    (target / 'report.md').unlink()
    imported = merge(target, [source])
    assert [item['seed'] for item in imported] == [2]
    assert (target / 'seed_2.json').read_bytes() == (source / 'seed_2.json').read_bytes()
    assert run(target, [1, 2], 1, resume=True)['num_seeds'] == 2
    with pytest.raises(FileExistsError):
        merge(target, [source])
