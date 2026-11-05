"""Explicit evaluation backend and conservative local inference source closure."""

import ast
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform

BACKENDS = ('official', 'bitboard')
ROOT = Path(__file__).resolve().parents[2]


def backend_name(value='official'):
    if value not in BACKENDS:
        raise ValueError('Unsupported evaluation feature backend: '+str(value))
    return value


def evaluation_runtime_identity(diagnostics=False):
    return dict(runtime_versions={name:version(name) for name in
                ['torch','numpy','gymnasium','stable-baselines3','sb3-contrib','cloudpickle']},
                python_version=platform.python_version(), diagnostics=diagnostics)


def inference_source_identity(resumable=False, root=ROOT):
    """Hash all statically imported local modules plus the checkpoint policy.

    Scan imports inside functions too, and both sides of conditional branches.
    SB3 loads CandidateScoringPolicy dynamically from the checkpoint, so its
    module is an explicit root. External packages remain in runtime identity.
    No numerical libraries are imported to compute this identity.
    """
    root = Path(root)
    pending = []
    files = set()

    def add_module(module):
        parts = module.split('.')
        if parts[0] != 'training':
            return
        for index in range(1, len(parts)+1):
            initializer = root.joinpath(*parts[:index], '__init__.py')
            if initializer.is_file() and initializer not in files:
                files.add(initializer);pending.append(initializer)
        path = root.joinpath(*parts).with_suffix('.py')
        if path.is_file() and path not in files:
            files.add(path);pending.append(path)

    roots = ['training.evaluation.evaluate', 'training.candidate_policy']
    if resumable:
        roots.append('training.evaluation.evaluate_resumable')
    for module in roots:
        path = root.joinpath(*module.split('.')).with_suffix('.py')
        if not path.is_file():
            raise FileNotFoundError('Missing inference root: '+module)
        add_module(module)
    while pending:
        path = pending.pop()
        relative = path.relative_to(root)
        package = list(relative.parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(), filename=str(relative))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        raise ValueError('Import escapes inference package: '+str(relative))
                    base = package[:len(package)-node.level+1]
                    if node.module:
                        base += node.module.split('.')
                    module = '.'.join(base)
                else:
                    module = node.module or ''
                if module.startswith('training'):
                    add_module(module)
                    for alias in node.names:
                        if alias.name != '*':
                            add_module(module+'.'+alias.name)
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}
