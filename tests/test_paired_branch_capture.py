import copy
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import paired_branch_capture as branch


def selection_data():
    return {'preferences': [dict(seed=seed,step=step,state_id=i*2+j,action=3,
        baseline_action=5,public_hash='public',snapshot_sha256='snapshot',label='beneficial',sources=['ppo_top_3'])
        for i,seed in enumerate(branch.SEEDS) for j,step in enumerate((300,100))]}


def test_selection_and_independent_branch_ids_preserve_paired_seed():
    cases = branch.selected_cases(selection_data())
    assert [c['seed'] for c in cases] == branch.SEEDS
    assert [c['step'] for c in cases] == [100]*4
    specs = branch.specs(cases)
    assert len({s['branch_id'] for s in specs}) == 8
    assert len({s['case']['seed'] for s in specs}) == 4
    assert [s['first_action'] for s in specs] == [5,3]*4


def test_missing_or_unsupported_preference_cannot_replace_selected_seed():
    prefs = selection_data()
    prefs['preferences'] = [p for p in prefs['preferences'] if p['seed'] != branch.SEEDS[0]]
    with pytest.raises(ValueError,match='predetermined'):branch.selected_cases(prefs)


@pytest.mark.parametrize('requested', [0,3,True,1.5])
def test_phase_chunk_budget_has_no_unbounded_or_boolean_option(requested):
    with pytest.raises(ValueError):branch.phase_plan(None,requested)


def test_phase_budget_clips_terminal_and_short_last_chunk():
    assert branch.phase_plan(None,2) == dict(chunks=2,maximum_new_environment_steps=4096)
    snap = dict(round=4992,workers=[{'active':True}]*5+[{'active':False}]*3)
    assert branch.phase_plan(snap,2) == dict(chunks=1,maximum_new_environment_steps=40)
    snap['workers'] = [{'active':False}]*8
    assert branch.phase_plan(snap,2)['chunks'] == 0


def test_checkpoint_boundary_inspection_never_reads_pickle(tmp_path,monkeypatch):
    assert branch.stored_boundary(tmp_path) == branch.boundary(None,None,0)
    folder = tmp_path/'chunk_0000';folder.mkdir()
    (folder/'state.pkl').write_bytes(b'not a pickle')
    (folder/'manifest.json').write_text(json.dumps(dict(round=0,environment_steps=0)))
    monkeypatch.setattr(pickle,'loads',lambda *a:pytest.fail('Unsafe state deserialization'))
    assert branch.stored_boundary(tmp_path)['chunks'] == 1


def test_checkpoint_gap_rejected_before_state_loading(tmp_path):
    (tmp_path/'chunk_0001').mkdir()
    with pytest.raises(ValueError,match='sequence'):branch.stored_boundary(tmp_path)


def context(env):
    env.reset(seed=992000)
    state = env.get_state();public = state['core'].get_public_observation()
    action = int(np.flatnonzero(env.action_masks())[0])
    case = dict(seed=992000,step=0,public_hash=branch.public_hash(public))
    spec = dict(case=case,branch_id='pair_base',first_action=action,horizon=2)
    return state,spec


def test_first_action_stop_and_restore_preserve_all_core_and_rng():
    env = branch.PairedCompactEnv(max_pieces=100000)
    restored = branch.PairedCompactEnv(max_pieces=100000)
    try:
        state,spec = context(env)
        view = env.paired_install(spec,state)
        assert branch.identical(env.paired_export()['state'],state)
        with pytest.raises(ValueError,match='intervention'):
            env.paired_step_map({'pair_base':(spec['first_action']+1)%1840})
        for offset in range(2):
            result = env.paired_step_map({'pair_base':int(np.flatnonzero(view['mask'])[0])})
            view = result['view']
        saved = env.paired_export()
        assert saved['steps'] == 2 and not saved['active'] and not saved['state']['finished']
        assert view is None and not result['row']['truncated']  # Artificial branch window, not game cap.
        assert restored.paired_install(spec,saved['state'],saved) is None
        assert branch.identical(saved,restored.paired_export())
        assert restored.paired_step_map({}) is None
        assert branch.identical(saved,restored.paired_export())
    finally:env.close();restored.close()


def test_shared_seed_dispatch_uses_branch_id_and_rejects_wrong_saved_identity():
    env = branch.PairedCompactEnv(max_pieces=100000)
    try:
        state,spec = context(env)
        env.paired_install(spec,state)
        with pytest.raises(ValueError,match='independent'):
            env.paired_step_map({992000:spec['first_action']})
        saved = env.paired_export();saved['branch_id'] = 'other'
        with pytest.raises(ValueError,match='complete saved'):
            env.paired_install(spec,state,saved)
    finally:env.close()


def sealed_context(tmp_path):
    env = branch.PairedCompactEnv(max_pieces=100000)
    try:state,spec = context(env)
    finally:env.close()
    folder = tmp_path/'counterfactual_states';folder.mkdir()
    path = folder/'seed_992000_step_0.pkl.gz'
    path.write_bytes(gzip.compress(pickle.dumps(state)))
    digest = branch.sha(path)
    meta = dict(seed=992000,step=0,sha256=digest,public_hash=spec['case']['public_hash'],rng_state=state['core'].rng.state)
    mp = path.with_suffix('.json');mp.write_text(json.dumps(meta))
    spec['case']['snapshot_sha256'] = digest
    design = dict(source_root=str(tmp_path),frozen_files={str(p):branch.sha(p) for p in (path,mp)})
    return spec['case'],design,state,path,mp


def test_complete_legacy_context_can_be_recovered_without_reset_or_rng_synthesis(tmp_path):
    case,design,state,_,_ = sealed_context(tmp_path)
    assert branch.identical(branch.load_start(case,design),state)


@pytest.mark.parametrize('invalid',['valid_legacy','missing_gameplay_rng','different_reward'])
def test_only_verified_legacy_reward_metadata_can_be_added(tmp_path,invalid):
    case,design,state,path,meta_path = sealed_context(tmp_path)
    state.pop('reward_profile')
    if invalid == 'missing_gameplay_rng':state.pop('numpy_generator')
    elif invalid == 'different_reward':state['reward_version']='unknown-reward'
    path.write_bytes(gzip.compress(pickle.dumps(state)));digest=branch.sha(path)
    case['snapshot_sha256']=digest
    meta=json.loads(meta_path.read_text());meta['sha256']=digest;meta_path.write_text(json.dumps(meta))
    design['frozen_files']={str(p):branch.sha(p) for p in (path,meta_path)}
    if invalid != 'valid_legacy':
        with pytest.raises(ValueError):branch.load_start(case,design)
    else:
        restored=branch.load_start(case,design)
        assert restored['reward_profile']=='default'
        assert branch.identical({k:v for k,v in restored.items() if k!='reward_profile'},state)


@pytest.mark.parametrize('damage',['external','legacy','private_rng','missing_rng'])
def test_legacy_integrity_or_rng_gaps_fail_closed(tmp_path,damage):
    case,design,state,path,meta_path = sealed_context(tmp_path)
    if damage == 'external':path.write_bytes(b'changed')
    elif damage == 'legacy':
        meta=json.loads(meta_path.read_text());meta['sha256']='wrong';meta_path.write_text(json.dumps(meta))
        design['frozen_files'][str(meta_path)] = branch.sha(meta_path)
    else:
        if damage == 'private_rng':state['core'].rng.state += 1
        else:state.pop('action_generator')
        path.write_bytes(gzip.compress(pickle.dumps(state)))
        new_sha = branch.sha(path);case['snapshot_sha256'] = new_sha
        meta=json.loads(meta_path.read_text());meta['sha256']=new_sha;meta_path.write_text(json.dumps(meta))
        design['frozen_files'] = {str(p):branch.sha(p) for p in (path,meta_path)}
    with pytest.raises(ValueError):branch.load_start(case,design)
