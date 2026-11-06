"""Own-policy capture preserves seeds, stopped lanes, state and real probability mode."""
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
from own_policy_capture import CaptureCompactEnv, greedy_details, read_chain, save_commit, sha
from verify_feature_backend_benchmark import identical
from training.tetris_core.pieces import make_piece


def test_explicit_seed_terminal_and_restore_do_not_reset():
    env = CaptureCompactEnv(max_pieces=2)
    restored = CaptureCompactEnv(max_pieces=2)
    try:
        view = env.capture_reset(100005)
        initial = env.capture_export()
        assert initial['state']['episode_seed'] == 100005 and initial['state']['pieces'] == 0
        for _ in range(2):
            outcome = env.capture_step(int(np.flatnonzero(view['mask'])[0]))
            view = outcome['view']
        assert view is None and outcome['row']['truncated'] and not outcome['row']['terminated']
        saved = env.capture_export()
        assert restored.capture_restore(saved) is None
        assert identical(saved, restored.capture_export())
        assert restored.capture_step(None) is None
        assert identical(saved, restored.capture_export())
        invalid = copy.deepcopy(saved); invalid['active'] = True
        with pytest.raises(ValueError):
            restored.capture_restore(invalid)
    finally:
        env.close(); restored.close()


def test_true_death_stays_stopped_without_reset():
    env = CaptureCompactEnv(max_pieces=20)
    try:
        env.capture_reset(100002)
        env.core.current = make_piece('I'); env.core.current['y'] = -2
        for x in range(3, 7):
            env.core.board[0][x] = 'Z'
        env._invalidate_candidates()
        placement = next(p for p in env.core.get_legal_placements()
                         if all(c['y'] < 0 for c in p['occupiedCells']))
        result = env.capture_step(placement['actionId'])
        assert result['row']['terminated'] and not result['row']['truncated']
        assert result['view'] is None and env.capture_step(None) is None
        assert env.episode_count == 1 and env.pieces == 1
    finally:
        env.close()


def test_details_use_actual_probability_mode_in_a_float32_tie():
    import torch
    class Policy:
        device = 'cpu'
        def get_distribution(self, obs, action_masks):
            probs = torch.tensor([[.5, .5, 0.]])
            logits = torch.tensor([[0., 1e-8, -1e8]])
            return SimpleNamespace(distribution=SimpleNamespace(probs=probs, logits=logits),
                get_actions=lambda deterministic: probs.argmax(dim=1))
    result = greedy_details(Policy(), {'state': np.zeros((1, 237), np.float32)},
                            np.array([[True, True, False]]), executed=np.array([1]))
    assert result['action'].tolist() == [0]
    assert result['margin'][0] > 0 and result['probability_of_executed'][0] == .5


def test_real_policy_details_and_shadow_do_not_change_actions_weights_or_rng():
    import torch
    from training.candidate_policy import CandidateScoringPolicy
    env = CaptureCompactEnv(max_pieces=20)
    try:
        view = env.capture_reset(100000)
        policy = CandidateScoringPolicy(env.observation_space, env.action_space, lambda _: 3e-4)
        policy.set_training_mode(False)
        obs = {key: value[None] for key, value in view['observation'].items()}
        mask = view['mask'][None]
        weights = copy.deepcopy(policy.state_dict()); rng = torch.get_rng_state().clone()
        action, _ = policy.predict(obs, deterministic=True, action_masks=mask)
        details = greedy_details(policy, obs, mask)
        shadow = greedy_details(policy, obs, mask, action)
        assert np.array_equal(action, details['action']) and np.array_equal(action, shadow['action'])
        assert identical(weights, policy.state_dict()) and torch.equal(rng, torch.get_rng_state())
        assert identical(env.capture_export()['state']['core'], env.core)
    finally:
        env.close()


def test_real_worker_restart_preserves_complete_state_and_stopped_lane(monkeypatch):
    from training.vector_env import make_vector_env
    def first_legal(policy, observations, masks, executed=None):
        actions = masks.argmax(axis=1)
        return dict(action=actions, probability=np.ones(len(actions)), margin=np.ones(len(actions)),
                    probability_of_executed=np.ones(len(actions)))
    monkeypatch.setattr(capture, 'greedy_details', first_legal)
    model = SimpleNamespace(policy=None)
    def vector():
        return make_vector_env(2, base_seed=100000, max_pieces=5, env_class=CaptureCompactEnv)
    env = vector()
    try:
        views = [env.env_method('capture_reset', 100000+i, indices=[i])[0] for i in range(2)]
        # A finished lane is retained while another lane continues.
        finished = env.env_method('capture_export', indices=[1])[0]
        finished['state']['pieces'] = 5; finished['state']['finished'] = True; finished['active'] = False
        views[1] = env.env_method('capture_restore', finished, indices=[1])[0]
        capture.advance_round(env, views, model, model, 0)
        saved = env.env_method('capture_export')
        expected_rows = capture.advance_round(env, views, model, model, 1)
        expected_state = env.env_method('capture_export')
    finally:
        env.close()
    env = vector()
    try:
        views = [env.env_method('capture_restore', state, indices=[i])[0] for i, state in enumerate(saved)]
        assert identical(saved, env.env_method('capture_export'))
        rows = capture.advance_round(env, views, model, model, 1)
        assert identical(rows, expected_rows) and identical(expected_state, env.env_method('capture_export'))
        assert [row['lane'] for row in rows] == [0]
        for round_index in range(2, 5):
            capture.advance_round(env, views, model, model, round_index)
        assert views == [None, None]
        assert env.env_method('capture_export', indices=[1])[0]['state']['pieces'] == 5
    finally:
        env.close()


def test_chain_hashes_before_pickle_rejects_identity_corruption_and_overwrite(tmp_path, monkeypatch):
    identity = dict(chunk_rounds=256, design_sha256='frozen')
    initial = dict(round=0, environment_steps=0, workers=[])
    previous = save_commit(tmp_path, 0, identity, None, initial, [], {})
    snapshot = dict(round=2, environment_steps=4, workers=[])
    rows = [dict(lane=i%2, action=i, features=np.arange(16, dtype=np.uint8)) for i in range(4)]
    save_commit(tmp_path, 1, identity, previous, snapshot, rows, {})
    restored, digest, count = read_chain(tmp_path, identity)
    assert restored == snapshot and count == 2 and digest == sha(tmp_path/'chunk_0001/manifest.json')
    with np.load(tmp_path/'chunk_0001/rows.npz', allow_pickle=False) as trace:
        assert trace['features'].shape == (4, 16)
    with pytest.raises(FileExistsError):
        save_commit(tmp_path, 1, identity, previous, snapshot, rows, {})
    with pytest.raises(ValueError):
        read_chain(tmp_path, dict(identity, design_sha256='wrong'))
    (tmp_path/'chunk_0000/state.pkl').write_bytes(b'bad')
    monkeypatch.setattr(pickle, 'loads', lambda _: pytest.fail('Unpickle happened before all hashes checked'))
    with pytest.raises(ValueError):
        read_chain(tmp_path, identity)


def test_chain_rejects_gap_and_incorrect_work_counters(tmp_path):
    identity = dict(chunk_rounds=256)
    previous = save_commit(tmp_path, 0, identity, None, dict(round=0, environment_steps=0), [], {})
    save_commit(tmp_path, 2, identity, previous, dict(round=2, environment_steps=2), [{'action': 1}], {})
    with pytest.raises(ValueError):
        read_chain(tmp_path, identity)
    (tmp_path/'chunk_0002').rename(tmp_path/'chunk_0001')
    manifest_path = tmp_path/'chunk_0001/manifest.json'
    manifest = json.loads(manifest_path.read_text()); manifest['number'] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='accounting'):
        read_chain(tmp_path, identity)
