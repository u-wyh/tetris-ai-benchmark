import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_head_direction as audit


def test_bias_cancels_and_affine_crossing_matches_direct_scores():
    h = np.array([[2., 1.], [0., 3.]])
    w, g, bias = np.array([3., 2.]), np.array([-2., 1.]), 71.
    delta = h[0]-h[1]
    margin, slope = delta@w, delta@g
    t, _, _ = audit.normal_crossing([margin], [slope], 0.)
    assert t == pytest.approx(1/3)
    for step in (0., t, t+1.):
        direct = h@(w+step*g)+bias
        assert direct[0]-direct[1] == pytest.approx(margin+step*slope)
    assert (h@(w+(t+1)*g)+bias)[1] > (h@(w+(t+1)*g)+bias)[0]


@pytest.mark.parametrize('margin,slope,tolerance,expected', [
    (2., -4., 0., .5), (2., 0., 0., np.inf), (2., 1., 0., np.inf),
    (0., 2., 0., 0.), (-.01, 2., 0., 0.), (1e-7, -1., 1e-6, 0.)])
def test_normal_crossings_preserve_ties_and_nonpositive_margins(margin,slope,tolerance,expected):
    assert audit.normal_crossing([margin], [slope], tolerance)[0] == expected


@pytest.mark.parametrize('margin,slope,expected', [
    (-2., 4., .5), (-2., 0., np.inf), (-2., -1., np.inf), (0., 1., 0.),
    (0., 0., np.inf), (2., -1., 0.)])
def test_pairwise_crossing_is_strictly_after_boundary(margin,slope,expected):
    assert audit.preference_crossing(margin,slope) == expected
    if margin <= 0 and np.isfinite(expected):
        assert margin+expected*slope == pytest.approx(0.)
        assert margin+(expected+.01)*slope > 0


def test_episode_weighting_does_not_weight_many_correlated_states_more():
    g = audit.equal_episode_direction([[2.,0.]]*5+[[0.,4.]], [0.]*6, [1]*5+[2])
    assert np.allclose(g, [.5,1.])
    assert not np.allclose(g, np.mean(np.array([[2.,0.]]*5+[[0.,4.]])*.5, axis=0))


def test_temperature_one_stability_and_zero_direction():
    assert np.isfinite(audit.equal_episode_direction([[1.],[1.]], [-10000.,10000.], [1,2])).all()
    assert np.array_equal(audit.equal_episode_direction([[1.],[-1.]], [0.,0.], [1,2]), [0.])


def test_seed_split_is_fixed_before_labels_and_separates_whole_episodes():
    for seed in [992000, 992010, 992030]:
        expected = int(hashlib.sha256(f'topk-head-geometry-v1:{seed}'.encode()).hexdigest(),16)%4 == 0
        assert audit.validation_seed(seed) is expected


def test_crossing_distribution_includes_infinity_and_zero_without_nan():
    result = audit.distribution([0.,2.,np.inf])
    assert result['zero'] == result['infinite'] == 1
    assert result['quantiles']['1.0'] == 'infinity'
    json.dumps(result,allow_nan=False)


def fixture_public(folder):
    states = [dict(state_id=i,seed=992000,step=i,baseline_action=0,public_hash='public',
                   snapshot_sha256='snapshot') for i in range(120)]
    pairs = [{**states[i//2], 'action':i%2+1,'horizon':5000,'sources':['ppo_top_3'],
              'label':'beneficial'} for i in range(177)]
    np.savez(folder/'public_inputs.npz', observation_state=np.zeros((120,237),np.float32),
        observation_candidate=np.zeros((120,1840,16),np.uint8), action_mask=np.ones((120,1840),bool),
        state_id=np.arange(120,dtype=np.int64))
    (folder/'preferences.json').write_text(json.dumps(dict(states=states,preferences=pairs,conflicting_public_contexts=[])))
    metadata = dict(status='passed',states=120,eligible_for_training=False,
                    per_branch_model_identity_complete=False,
                    artifacts={n:audit.sha(folder/n) for n in ('preferences.json','public_inputs.npz')})
    (folder/'evidence_audit.json').write_text(json.dumps(metadata))
    return {str(p):audit.sha(p) for p in folder.iterdir()}


def reseal(folder):
    path = folder/'evidence_audit.json'
    metadata = json.loads(path.read_text())
    metadata['artifacts'] = {n:audit.sha(folder/n) for n in ('preferences.json','public_inputs.npz')}
    path.write_text(json.dumps(metadata))
    return {str(p):audit.sha(p) for p in folder.iterdir()}


def test_public_loader_accepts_fixed_177_but_rejects_unsealed_bytes(tmp_path):
    frozen = fixture_public(tmp_path)
    assert len(audit.load_public(tmp_path,frozen)[2]) == 177
    (tmp_path/'preferences.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        audit.load_public(tmp_path,frozen)


@pytest.mark.parametrize('change', ['hidden_input','state_order','source','identity','illegal','promoted'])
def test_public_schema_and_provenance_cannot_silently_change(tmp_path,change):
    fixture_public(tmp_path)
    path = tmp_path/'preferences.json'
    preferences = json.loads(path.read_text())
    if change == 'state_order':preferences['states'].reverse()
    elif change == 'source':preferences['preferences'][0]['sources'] = ['bc']
    elif change == 'identity':preferences['preferences'][0]['seed'] = 777
    elif change == 'illegal':preferences['preferences'][0]['action'] = 1840
    elif change == 'hidden_input':
        archive = tmp_path/'public_inputs.npz'
        with np.load(archive) as z:arrays = {k:z[k] for k in z.files}
        arrays['hidden_rng'] = np.zeros(120)
        np.savez(archive,**arrays)
    else:
        audit_path = tmp_path/'evidence_audit.json'
        meta = json.loads(audit_path.read_text());meta['eligible_for_training'] = True
        audit_path.write_text(json.dumps(meta))
    path.write_text(json.dumps(preferences))
    with pytest.raises(ValueError):audit.load_public(tmp_path,reseal(tmp_path))


def test_source_hash_change_refused_before_any_model_load(tmp_path):
    path = tmp_path/'source';path.write_text('old')
    frozen = {str(path):audit.sha(path)}
    path.write_text('new')
    with pytest.raises(ValueError,match='Frozen geometry input changed'):audit.check_frozen(frozen)
