import json

import pytest

from training.evaluation.evaluate_resumable import collect_cached_rows


def identity():
    return dict(seeds=[12,11,13], max_pieces=20, model_sha256='frozen',
                evaluation_source_sha256={'evaluator':'frozen'},diagnostics=False,
                protocol='development',committed_steps=4096)


def row(seed):
    return dict(seed=seed,pieces_survived=20,lines=8,score=100,
                episode_reward=10.0,game_over=False,survived_cap=True)


def test_interrupted_evaluation_reuses_completed_seeds_and_preserves_frozen_order(tmp_path):
    def interrupted(pending):
        assert pending == [12,11,13]
        yield row(11)
        raise RuntimeError('worker interrupted')
    with pytest.raises(RuntimeError,match='interrupted'):
        collect_cached_rows(tmp_path,identity(),interrupted)
    assert (tmp_path/'seed_11.json').exists()
    def resume(pending):
        assert pending == [12,13]
        yield row(13);yield row(12)
    rows,reused=collect_cached_rows(tmp_path,identity(),resume)
    assert [r['seed'] for r in rows]==[12,11,13] and reused==[11]
    def never(pending):raise AssertionError('Completed seeds must not be rerun')
    assert collect_cached_rows(tmp_path,identity(),never)[0]==rows


@pytest.mark.parametrize('key,value',[('model_sha256','other'),('max_pieces',21),
    ('seeds',[11,12,13]),('evaluation_source_sha256',{'evaluator':'changed'}),
    ('diagnostics',True),('committed_steps',8192),('protocol','final')])
def test_cache_cannot_cross_model_source_protocol_seed_or_cap_identity(tmp_path,key,value):
    collect_cached_rows(tmp_path,identity(),lambda seeds:map(row,seeds))
    changed={**identity(),key:value}
    with pytest.raises(ValueError,match='identity differs'):
        collect_cached_rows(tmp_path,changed,lambda seeds:pytest.fail('Must reject before computing'))


def test_corruption_is_rejected_instead_of_silently_recomputed(tmp_path):
    collect_cached_rows(tmp_path,identity(),lambda seeds:map(row,seeds))
    path=tmp_path/'seed_11.json';record=json.loads(path.read_text());record['row']['score']=999
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='integrity'):
        collect_cached_rows(tmp_path,identity(),lambda seeds:pytest.fail('Corrupt evidence must be preserved'))


def test_invalid_or_missing_worker_results_never_become_complete(tmp_path):
    with pytest.raises(ValueError,match='all missing'):
        collect_cached_rows(tmp_path,identity(),lambda seeds:[])
    with pytest.raises(ValueError,match='terminal'):
        collect_cached_rows(tmp_path,identity(),lambda seeds:[{**row(12),'survived_cap':False}])
    assert not (tmp_path/'seed_12.json').exists()
