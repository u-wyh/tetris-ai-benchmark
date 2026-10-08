import copy
import json
from unittest.mock import patch

import pytest

from training.evaluation.traditional_v2 import play, run
from training.evaluation.merge_traditional import merge
from training.tetris_core import TetrisCore
from training.tetris_core.pieces import make_piece
from training.traditional.v2 import V2, V2Config, simulate_public


def comparable(predicted, actual):
    for key in ('board','currentPiece','hold','holdAvailable','score','lines','level','phase','gameOver'):
        assert predicted[key] == actual[key], key
    assert predicted['next'] == actual['next'][:len(predicted['next'])]


def test_simulation_matches_core_normal_empty_hold_and_swap():
    for seed in (123, 234):
        core = TetrisCore(seed)
        initial = core.get_public_observation()
        placements = core.get_legal_placements()
        for placement in (placements[0], next(p for p in placements if p['hold'])):
            independent = TetrisCore(seed)
            predicted, cleared = simulate_public(initial, placement)
            before = independent.lines
            actual = independent.step(placement['actionId'])
            comparable(predicted, actual)
            assert cleared == independent.lines-before
        core.step(next(p for p in placements if p['hold'])['actionId'])
        after_hold = core.get_public_observation()
        swap = next(p for p in core.get_legal_placements() if p['hold'])
        predicted, _ = simulate_public(after_hold, swap)
        comparable(predicted, core.step(swap['actionId']))


def test_public_boundary_determinism_and_no_hidden_queue():
    core = TetrisCore(12345)
    public = core.get_public_observation()
    before = copy.deepcopy(public)
    policy = V2()
    first = policy.choose(public)
    assert first == policy.choose(public)
    assert first in {p['actionId'] for p in core.get_legal_placements()}
    assert public == before
    core.queue[3:] = ['I','I']
    core.bag.reverse()
    core.rng.state = 0
    assert first == policy.choose(core.get_public_observation())
    assert policy.last_timing_ms['future_legal'] >= 0


def test_visible_next_consumption_and_unknown_termination():
    core = TetrisCore(7)
    public = core.get_public_observation()
    held = next(p for p in core.get_legal_placements() if p['hold'])
    predicted, _ = simulate_public(public, held)
    assert predicted['hold'] == public['currentPiece']['type']
    assert predicted['currentPiece']['type'] == public['next'][1]
    assert predicted['next'] == public['next'][2:]
    second = next(p for p in core.get_legal_placements() if not p['hold'])
    # A single visible current and no Next must stop after placement.
    short = {**public, 'next': []}
    unknown, _ = simulate_public(short, second)
    assert unknown['phase'] == 'unknown' and unknown['currentPiece'] is None
    assert V2().choose(public) == V2().choose(public)


def test_top_out_and_doom_penalty():
    core = TetrisCore(1)
    core.board = [['T']*10 for _ in range(20)]
    core.current = make_piece('I')
    core.current['y'] = -2
    public = core.get_public_observation()
    candidate = core.get_legal_placements()[0]
    predicted, cleared = simulate_public(public, candidate)
    assert cleared == 0 and predicted['gameOver']
    comparable(predicted, core.step(candidate['actionId']))


def test_modes_hold_and_reproducible_rollout():
    for mode in ('heuristic','hold','beam'):
        config = V2Config(mode=mode,beam_width=4)
        first, second = play(19, 12, config), play(19, 12, config)
        assert (first['actions'],first['score'],first['lines']) == (second['actions'],second['score'],second['lines'])
        assert first['truncated'] and len(first['decision_ms']) == 12
        if mode == 'heuristic':
            assert first['hold_actions'] == 0


def test_benchmark_resume_and_config_guard(tmp_path):
    output = tmp_path/'v2'
    config = V2Config(mode='beam',beam_width=2)
    real_play = play
    def broken(seed, cap, policy, progress):
        if seed == 2:
            raise RuntimeError('injected')
        return real_play(seed,cap,policy,progress)
    with patch('training.evaluation.traditional_v2.play',side_effect=broken):
        with pytest.raises(RuntimeError,match='injected'):
            run(output,[1,2],1,config)
    saved = (output/'seed_1.json').read_bytes()
    assert list(output.glob('error_*.json'))
    assert run(output,[1,2],1,config,resume=True)['num_seeds'] == 2
    assert saved == (output/'seed_1.json').read_bytes()
    with pytest.raises(ValueError,match='mismatch'):
        run(output,[1,2],1,V2Config(mode='hold'),resume=True)
    with pytest.raises(FileExistsError):
        run(output,[1,2],1,config)


def test_v2_compatible_shard_merge(tmp_path):
    target, shard = tmp_path/'target', tmp_path/'shard'
    config = V2Config(mode='beam',beam_width=2)
    run(target,[1],1,config)
    run(shard,[2],1,config)
    imported = merge(target,[shard])
    assert [item['seed'] for item in imported] == [2]
    assert (target/'seed_2.json').read_bytes() == (shard/'seed_2.json').read_bytes()
    assert run(target,[1,2],1,config,resume=True)['num_seeds'] == 2
