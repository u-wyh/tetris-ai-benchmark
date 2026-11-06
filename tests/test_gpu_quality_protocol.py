import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_gpu_quality import matching_protocol


def pair():
    left = dict(training_sources={'a': 'train'}, inference_sources={'b': 'eval'},
        evaluation_runtime={'python_version': '3'}, native_feature_runtime={'sha256': 'native'},
        seeds=list(range(100000, 100008)), max_pieces=20000, chunk_rounds=256,
        device='cuda', n_envs=8, optimizer_steps=0,
        models=dict(candidate=dict(sha256='original', counts=[30003200, 36625]),
                    shadow=dict(sha256='four', counts=[4096, 36630])))
    right = copy.deepcopy(left)
    right['models'] = dict(candidate=copy.deepcopy(left['models']['shadow']),
                           shadow=copy.deepcopy(left['models']['candidate']))
    return left, right


def test_opposite_roles_same_numerical_protocol():
    matching_protocol(*pair())


@pytest.mark.parametrize('key,value', [('device', 'cpu'), ('n_envs', 2),
    ('inference_sources', {'b': 'changed'}), ('evaluation_runtime', {'python_version': 'different'}),
    ('native_feature_runtime', {'sha256': 'other'}), ('optimizer_steps', 1)])
def test_mixed_protocol_rejected(key, value):
    left, right = pair()
    right[key] = value
    with pytest.raises(ValueError, match='protocol'):
        matching_protocol(left, right)


def test_identical_cpu_protocol_still_rejected():
    left, right = pair()
    left['device'] = right['device'] = 'cpu'
    with pytest.raises(ValueError, match='GPU8'):
        matching_protocol(left, right)


def test_wrong_shadow_or_source_counters_rejected():
    left, right = pair()
    right['models']['shadow']['counts'][0] += 1
    with pytest.raises(ValueError, match='roles'):
        matching_protocol(left, right)
