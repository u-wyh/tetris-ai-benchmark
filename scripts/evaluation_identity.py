"""Shared, explicit evaluator selection for frozen development protocols."""

import hashlib
from importlib.metadata import version
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
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
    return {name:hashlib.sha256((ROOT/'training/evaluation'/name).read_bytes()).hexdigest()
            for name in ('evaluate.py','worker_budget.py')}


def runtime_identity():
    return dict(runtime_versions={name:version(name) for name in
                ['torch','numpy','gymnasium','stable-baselines3','sb3-contrib']},
                python_version=platform.python_version(), diagnostics=False)


def validate_evaluator(protocol, identity):
    module = module_for(protocol)
    if source_identity(module) != identity['evaluation_source_sha256']:
        raise RuntimeError('Frozen evaluator identity changed')
    if module == RESUMABLE and any(identity.get(k) != v for k,v in runtime_identity().items()):
        raise RuntimeError('Frozen evaluator runtime identity changed')
    return module
