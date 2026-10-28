import copy

import numpy as np
import pytest
import torch

import training.bc_sequence_supervised as supervised


def test_eligibility_requires_observed_future_and_full_contiguous_fragments(monkeypatch):
    monkeypatch.setattr(supervised,'FOLLOW',2)
    rows = [dict(clear=4 if i == 1 else 0) for i in range(6)]
    frames = [dict(piece=i) for i in range(6)]
    selected = supervised.select_native_frames(rows,frames,2,False)
    assert [r['piece'] for r in selected] == [0,1]
    # Death at the exact future horizon is not a survived demonstration.
    assert not supervised.select_native_frames(rows[:4],frames[:4],2,True)
    assert not supervised.select_native_frames(rows[:3],frames[:3],2,False)


def test_short_development_gate_rejects_early_death_or_insufficient_caps():
    a = dict(mean_pieces=20000,p10=20000,early=0,capped=16,games=16,
             high_fraction=0,new_holes_per_100=2)
    b = dict(a,capped=15)
    result = dict(summary=dict(bc=a,candidate=b),
        paired_to_bc=dict(candidate=dict(common_score_ratio=1.04,positive_seeds=12,ratio_ci95=[1.01,1.05])))
    assert supervised.development_gate(result)['passed']
    result['summary']['candidate']['early'] = 1
    assert not supervised.development_gate(result)['passed']
    result['summary']['candidate'].update(early=0,capped=14)
    assert not supervised.development_gate(result)['passed']


def test_real_gradients_optimizer_rng_save_load_and_terminal_stop(tmp_path,monkeypatch):
    if not supervised.BC.exists():
        pytest.skip('Frozen artifact unavailable')
    monkeypatch.setattr(supervised,'WORK',tmp_path)
    monkeypatch.setattr(supervised,'guard',lambda:None)
    monkeypatch.setattr(supervised,'operation',lambda message:None)
    monkeypatch.setattr(supervised,'learning_checks',lambda a,b:dict(bc_agreement=False,native_signal=True,target_signal=True))
    torch.set_num_threads(1)
    env = supervised.study.LongCandidateTetrisEnv(max_pieces=16)
    env.reset(seed=93)
    sample = supervised.public_sample(env)
    original = supervised.MaskablePPO.load(supervised.BC,device='cpu')
    action = supervised.study.fast_action(original,env.get_observation(),env.action_masks())
    other = next(int(i) for i in sample['legal_ids'] if i != action)
    parts = {partition:{source:[dict(sample,action=action if source=='bc' else other,
        episode=f'{partition}:{source}',fragment=0) for _ in range(32)]
        for source in ('bc','ppo','transfer')} for partition in ('train','validation')}
    result = supervised.train(parts,dict(device='cpu'),'unit-test')
    assert result is None
    smoke = supervised.read_record(tmp_path/'gradient_save_load.json')
    assert smoke['passed'] and smoke['real_updates'] == 2
    assert set(smoke['optimizer_steps']) == {2}
    checkpoint = tmp_path/'checkpoints'/'epoch_01'
    trained,optimizer,state = supervised.load_update(checkpoint,'cpu','unit-test')
    assert state['updates'] == 2
    assert any(not torch.equal(v,original.policy.state_dict()[name])
               for name,v in trained.policy.state_dict().items() if name.startswith('score_head.'))
    assert all(torch.equal(v,original.policy.value_net.state_dict()[name])
               for name,v in trained.policy.value_net.state_dict().items())
    # Resuming a semantic stop must not train another epoch.
    assert supervised.train(parts,dict(device='cpu'),'unit-test') is None
    assert supervised.read_record(tmp_path/'training_result.json')['updates'] == 2
    with pytest.raises(RuntimeError,match='identity'):
        supervised.load_update(checkpoint,'cpu','wrong-identity')
    env.close()
