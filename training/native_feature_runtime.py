"""Explicit native artifact declaration and pre-write checkpoint validation."""
import json
from pathlib import Path

from training.native_candidate_features import NativeCandidateFeatures


def load_runtime(spec):
    if (not isinstance(spec, dict) or set(spec) != {'manifest', 'sha256'} or
            not isinstance(spec['manifest'], str) or not Path(spec['manifest']).is_absolute() or
            not isinstance(spec['sha256'], str) or len(spec['sha256']) != 64 or
            any(c not in '0123456789abcdef' for c in spec['sha256'])):
        raise ValueError('Native feature runtime requires absolute manifest and SHA256')
    return NativeCandidateFeatures(spec['manifest'], expected_manifest_sha256=spec['sha256'])


def runtime_identity(config):
    spec = config.get('native_feature_runtime')
    native = config.get('candidate_feature_backend') == 'bitboard_native'
    if spec is None:
        if native:
            raise ValueError('Native features require an explicit pinned runtime')
        return None
    # A bitboard control arm may explicitly declare the same runtime without
    # using C features. Ordinary backends with no declaration load no binary.
    if (config.get('candidate_feature_backend') not in ('bitboard', 'bitboard_native') or
            config.get('candidate_search_backend') != 'bitboard_compact' or
            not config.get('candidate_policy') or not config.get('fast_candidate_env') or
            config.get('failure_curriculum')):
        raise ValueError('Native runtime declaration requires ordinary compact Candidate features')
    loaded = load_runtime(spec)
    return dict(declaration=spec, artifact=loaded.identity)


def verify_runtime_identity(config, metadata):
    identity = runtime_identity(config)
    if config.get('native_feature_runtime_identity') != identity:
        raise RuntimeError('Run native feature runtime and config identity disagree')
    if metadata.get('native_feature_runtime_identity') != identity:
        raise RuntimeError('Run native feature runtime and metadata disagree')
    if ('candidate_feature_backend' in metadata and
            metadata['candidate_feature_backend'] != config.get('candidate_feature_backend', 'official')):
        raise RuntimeError('Run Candidate feature backend and metadata disagree')
    return identity


def preflight(run_dir, resume, config_file):
    """Read only: reject changed runtime or requested backend before lock/log writes."""
    preset = json.loads(Path(config_file).read_text()) if config_file else None
    if not resume:
        return runtime_identity(preset or {})
    config = json.loads((run_dir/'config.json').read_text())
    metadata = json.loads((run_dir/'metadata.json').read_text())
    if preset is not None:
        if preset.get('candidate_feature_backend', 'official') != config.get('candidate_feature_backend', 'official'):
            raise RuntimeError('Resume Candidate feature backend differs from recorded run')
        if preset.get('native_feature_runtime') != config.get('native_feature_runtime'):
            raise RuntimeError('Resume native feature runtime differs from recorded run')
    return verify_runtime_identity(config, metadata)
