"""Hash-first loading and exact replay across a real worker-process boundary."""
import copy
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import own_policy_capture as capture
from own_policy_capture import CaptureCompactEnv, sha
from restore_own_death_contexts import load_verified_chunk, restore_vector, replay_chunk
from restore_rank_contexts import observation_digest
from verify_feature_backend_benchmark import identical
from training.vector_env import make_vector_env
from training.tetris_core.pieces import make_piece


def test_verified_chunk_checks_external_manifest_identity_and_both_files_before_pickle(tmp_path, monkeypatch):
    identity = dict(source='frozen')
    state = dict(round=0, environment_steps=0)
    (tmp_path/'state.pkl').write_bytes(pickle.dumps(state, protocol=5))
    np.savez_compressed(tmp_path/'rows.npz')
    manifest = dict(identity=identity, round=0, environment_steps=0, rows=0,
                    files={name:sha(tmp_path/name) for name in ['state.pkl','rows.npz']})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    digest = sha(tmp_path/'manifest.json')
    assert load_verified_chunk(tmp_path, digest, identity)[0] == state
    monkeypatch.setattr(pickle, 'loads', lambda _:pytest.fail('Bad input reached pickle'))
    with pytest.raises(ValueError, match='manifest checksum'):
        load_verified_chunk(tmp_path, 'wrong', identity)
    with pytest.raises(ValueError, match='identity'):
        load_verified_chunk(tmp_path, digest, dict(source='other'))
    (tmp_path/'rows.npz').write_bytes(b'bad')
    with pytest.raises(ValueError, match='file checksum'):
        load_verified_chunk(tmp_path, digest, identity)


def test_replay_restores_actual_stopped_lane_and_full_rng_through_new_forkserver_workers(monkeypatch):
    def greedy(policy, observations, masks, executed=None):
        actions = masks.argmax(axis=1) if executed is None else masks.shape[1]-1-masks[:,::-1].argmax(axis=1)
        return dict(action=actions, probability=np.ones(len(actions)), margin=np.ones(len(actions)),
                    probability_of_executed=np.ones(len(actions)))
    monkeypatch.setattr(capture, 'greedy_details', greedy)
    model = SimpleNamespace(policy=None)
    def vector():
        return make_vector_env(2, max_pieces=3, env_class=CaptureCompactEnv)
    env = vector()
    try:
        views = [env.env_method('capture_reset', 100000+lane, indices=[lane])[0] for lane in range(2)]
        # True death in lane1 at round0; it must remain present and stopped on restart.
        saved = env.env_method('capture_export', indices=[1])[0]
        core = saved['state']['core']; core.current = make_piece('I'); core.current['y'] = -2
        for x in range(3,7):
            core.board[0][x] = 'Z'
        placement = next(p for p in core.get_legal_placements() if all(c['y'] < 0 for c in p['occupiedCells']))
        env.env_method('capture_restore', saved, indices=[1])
        outcome = env.env_method('capture_step', placement['actionId'], indices=[1])[0]
        assert outcome['row']['terminated']; views[1] = outcome['view']
        views[0] = env.env_method('capture_step', int(views[0]['mask'].argmax()), indices=[0])[0]['view']
        initial = dict(workers=env.env_method('capture_export'), round=1,
            views=[None if view is None else observation_digest(**view) for view in views])
        records = []
        for round_index in [1,2]:
            records.extend(capture.advance_round(env, views, model, model, round_index))
        expected = env.env_method('capture_export')
    finally:
        env.close()
    data = {key:np.asarray([row[key] for row in records]) for key in records[0]}
    target = dict(key='one', lane=0, round=1, candidate_action=int(data['action'][0]),
        shadow_action=int(data['shadow_action'][0]), observation_digest=initial['views'][0])
    env = vector()
    try:
        restored = restore_vector(env, initial)
        contexts, end_views, steps = replay_chunk(env, restored, data, 1, 3, [target])
        assert steps == 2 and end_views == [None,None]
        assert contexts['one']['state']['pieces'] == 1
        assert contexts['one']['state']['episode_count'] == 1
        assert identical(expected, env.env_method('capture_export'))
        assert identical(initial['workers'][1], expected[1])
        broken = copy.deepcopy(data); broken['mask_sha256'][0] = 'x'*64
        restored = restore_vector(env, initial)
        with pytest.raises(ValueError, match='observation, mask'):
            replay_chunk(env, restored, broken, 1, 3, [target])
        restored = restore_vector(env, initial)
        broken = copy.deepcopy(data); broken['lane'][0] = 1
        with pytest.raises(ValueError, match='ordered active'):
            replay_chunk(env, restored, broken, 1, 3, [target])
    finally:
        env.close()


def test_replay_budget_and_snapshot_width_are_rejected_before_progress():
    with pytest.raises(ValueError, match='one original bounded chunk'):
        replay_chunk(None, [], {}, 0, 257, [])
    env = SimpleNamespace(num_envs=8)
    with pytest.raises(ValueError, match='every lane'):
        restore_vector(env, dict(workers=[], views=[]))
