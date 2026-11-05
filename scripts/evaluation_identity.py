"""Shared, explicit evaluator selection for frozen development protocols."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
LEGACY = 'training.evaluation.evaluate'
RESUMABLE = 'training.evaluation.evaluate_resumable'


def module_for(protocol):
    module = protocol.get('evaluation_module', LEGACY)
    if module not in (LEGACY, RESUMABLE):
        raise ValueError('Unsupported evaluation module: ' + str(module))
    return module


def source_identity(module):
    if module == RESUMABLE:
        from training.evaluation.evaluate_resumable import source_identity as resumable_identity
        return resumable_identity()
    if module != LEGACY:
        raise ValueError('Unsupported evaluation module')
    from training.evaluation.feature_backend import inference_source_identity
    return inference_source_identity()


def runtime_identity():
    from training.evaluation.feature_backend import evaluation_runtime_identity
    return evaluation_runtime_identity()


def feature_backend(protocol):
    from training.evaluation.feature_backend import backend_name
    return backend_name(protocol.get('candidate_feature_backend','official'))


def validate_evaluator(protocol, identity):
    module = module_for(protocol)
    if source_identity(module) != identity['evaluation_source_sha256']:
        raise RuntimeError('Frozen evaluator identity changed')
    if feature_backend(protocol) != identity.get('candidate_feature_backend','official'):
        raise RuntimeError('Frozen evaluation feature backend differs')
    if module == RESUMABLE and any(identity.get(k) != v for k,v in runtime_identity().items()):
        raise RuntimeError('Frozen evaluator runtime identity changed')
    return module
