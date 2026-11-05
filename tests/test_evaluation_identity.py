from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluation_identity as identity


def test_legacy_protocol_defaults_and_runtime_independent_compatibility(monkeypatch):
    monkeypatch.setattr(identity,'source_identity',lambda module:{'evaluate.py':'fixed'})
    assert identity.validate_evaluator({},dict(evaluation_source_sha256={'evaluate.py':'fixed'}))==identity.LEGACY
    with pytest.raises(ValueError,match='Unsupported'):
        identity.module_for({'evaluation_module':'untrusted.module'})


def test_resumable_protocol_requires_matching_sources_and_runtime(monkeypatch):
    monkeypatch.setattr(identity,'source_identity',lambda module:{'all.py':'fixed'})
    runtime=dict(runtime_versions={'torch':'test'},python_version='3.test',diagnostics=False)
    monkeypatch.setattr(identity,'runtime_identity',lambda:runtime)
    protocol=dict(evaluation_module=identity.RESUMABLE)
    baseline=dict(evaluation_source_sha256={'all.py':'fixed'},**runtime)
    assert identity.validate_evaluator(protocol,baseline)==identity.RESUMABLE
    for key,value in [('python_version','changed'),('runtime_versions',{'torch':'changed'}),('diagnostics',True)]:
        with pytest.raises(RuntimeError,match='runtime'):
            identity.validate_evaluator(protocol,{**baseline,key:value})
    with pytest.raises(RuntimeError,match='identity changed'):
        identity.validate_evaluator(protocol,{**baseline,'evaluation_source_sha256':{}})
