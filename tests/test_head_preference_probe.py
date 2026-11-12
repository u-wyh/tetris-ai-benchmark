import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import head_preference_probe as probe
from test_head_representation import fixture


def synthetic():
    h = torch.zeros((8, 3, 2), dtype=torch.float64)
    h[:4, 1, 0] = 2.;h[4:, 1, 1] = 1.
    raw = torch.zeros((8, 3), dtype=torch.float64)
    mask = torch.ones((8, 3), dtype=torch.bool);mask[:, 2] = False
    return h, raw, mask, torch.ones(4, dtype=torch.int64), torch.zeros(4, dtype=torch.int64)


def test_zero_delta_teacher_kl_and_exact_pair_gradient():
    delta = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    terms = probe.loss_terms(delta, *synthetic())
    assert terms[1].item() == pytest.approx(np.log(2.))
    assert terms[2].item() == pytest.approx(0., abs=1e-12)
    terms[0].backward()
    assert torch.allclose(delta.grad, torch.tensor([-1., 0.], dtype=torch.float64))


def test_masked_padding_cannot_change_loss_or_gradient():
    args = synthetic();delta = torch.tensor([.1, .2], dtype=torch.float64, requires_grad=True)
    first = probe.loss_terms(delta, *args)[0];gradient = torch.autograd.grad(first, delta)[0]
    h, raw, mask, good, base = args
    h[:, 2] = 1e8;raw[:, 2] = 1e10
    second = probe.loss_terms(delta, h, raw, mask, good, base)[0]
    # Padded pair competitors are not scored; normal logits must mask them.
    assert second.item() == first.item()
    assert torch.equal(torch.autograd.grad(second, delta)[0], gradient)


def test_large_logits_have_finite_loss_and_gradient():
    args = synthetic();args[1][:4, 1] = -10000.
    delta = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    terms = probe.loss_terms(delta, *args)
    terms[0].backward()
    assert all(torch.isfinite(t) for t in terms) and torch.isfinite(delta.grad).all()


def test_whole_episode_training_schedule_never_selects_validation(tmp_path):
    arrays, metadata = fixture(tmp_path)
    selected = [probe.batch_indices(metadata, step) for step in range(256)]
    assert all(len(indices) == 8 for indices in selected)
    normal = {i for indices in selected for i in indices[4:]}
    assert len(normal) == 316
    assert {metadata['states'][i]['seed'] for i in normal} == set(range(100000, 100004))
    original = probe.pack(arrays, metadata, selected[0], 'cpu')
    for i, state in enumerate(metadata['states']):
        if state['split'] == 'validation':
            arrays['h'][arrays['offsets'][i]:arrays['offsets'][i+1]] = 12345.
    assert all(torch.equal(a, b) for a, b in zip(original, probe.pack(arrays, metadata, selected[0], 'cpu')))


def test_validation_tag_inside_training_slot_is_refused(tmp_path):
    _, metadata = fixture(tmp_path);metadata['states'][8]['split'] = 'validation'
    with pytest.raises(ValueError, match='Validation'):
        probe.batch_indices(metadata, 0)


def test_validation_metrics_equal_episodes_instead_of_overweighting_two_cases():
    assert probe.episode_mean([0., 2., 10.], [992011, 992011, 992016]) == pytest.approx(5.5)


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.score_head = torch.nn.Sequential(torch.nn.Linear(2, 1))
        self.critic = torch.nn.Linear(2, 1)

    def set_training_mode(self, enabled):
        self.train(enabled)


def test_apply_delta_changes_only_copied_final_weight_and_preserves_source():
    policy = ToyPolicy();before = copy.deepcopy(policy.state_dict())
    candidate = probe.apply_delta(policy, np.array([.25, -.5]))
    assert all(torch.equal(before[name], value) for name, value in policy.state_dict().items())
    assert [name for name, value in candidate.state_dict().items() if not torch.equal(before[name], value)] == ['score_head.0.weight']
    assert not candidate.training and not any(p.requires_grad for p in candidate.parameters())


def test_cuda_unavailable_refuses_fit_before_any_cpu_optimizer(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **k: pytest.fail('CPU optimizer fallback'))
    with pytest.raises(RuntimeError, match='CUDA required'):
        probe.fit(None, None, {}, None, None)


@pytest.mark.parametrize('corruption', ['none', 'wrong_model', 'not_current_positive', 'wrong_case'])
def test_only_matching_current_gpu_labels_pass_readiness(tmp_path, corruption):
    from test_paired_branch_validation import validation_data
    preferences = validation_data()
    for p in preferences['preferences']:
        if p['seed'] in [992011, 992016, 992027]:
            p['state_id'] += 50
    public = dict(observation_state=np.zeros((120, 237), np.float32),
        observation_candidate=np.zeros((120, 1840, 16), np.uint8), action_mask=np.ones((120, 1840), bool))
    states = [{} for _ in range(120)];design = {};evidence_paths = {}
    for split in ('train', 'validation'):
        cases = probe.selected_cases(preferences, split)
        for case in cases:
            states[case['state_id']] = {k: v for k, v in case.items() if k != 'action'}
        evidence = dict(status='passed', label_counts=dict(beneficial=4),
            model=dict(sha256=probe.MODEL_SHA, counts=[30003200, 36625]), case_split=split,
            per_branch_model_identity_explicit=True, optimizer_steps=0, comparisons=[dict(case=c) for c in cases])
        receipt = dict(status='probe_passed', source_full_state_restored=True,
            branch_ids=[s['branch_id'] for s in probe.specs(cases)],
            public_digests=[probe.observation_digest(**probe.public_case(public, c['state_id'])) for c in cases for _ in range(2)])
        for suffix, value in [('window', evidence), ('probe', receipt)]:
            path = tmp_path/(split+'_'+suffix+'.json');path.write_text(json.dumps(value));design[split+'_'+suffix] = str(path)
        evidence_paths[split] = evidence
    val = evidence_paths['validation']
    if corruption == 'wrong_model':val['model']['sha256'] = 'another_model'
    if corruption == 'not_current_positive':val['label_counts']['beneficial'] = 3
    if corruption == 'wrong_case':val['comparisons'][0]['case']['action'] += 99
    Path(design['validation_window']).write_text(json.dumps(val))
    if corruption == 'none':
        assert len(probe.current_cases(design, preferences, public, states)['validation']) == 4
    else:
        with pytest.raises(ValueError, match='readiness'):
            probe.current_cases(design, preferences, public, states)
