import copy
import importlib.util
from pathlib import Path


spec=importlib.util.spec_from_file_location('development_comparison',Path(__file__).resolve().parents[1]/'scripts/compare_development_evaluations.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def result():
    return dict(model_sha256='baseline',evaluation_source_sha256={'evaluate.py':'fixed','worker_budget.py':'fixed'},
                seeds=[1,2],num_seeds=2,max_pieces=5000,committed_steps=0,
                per_seed=[dict(seed=1,pieces_survived=5000,score=100,survived_cap=True),
                          dict(seed=2,pieces_survived=4000,score=80,survived_cap=False)])


def test_matched_comparison_is_paired_and_censoring_is_explicit():
    baseline=result();candidate=copy.deepcopy(baseline);candidate['model_sha256']='new'
    candidate['per_seed'][1].update(pieces_survived=4500,score=90)
    candidate['per_seed'].reverse()
    report=module.compare(baseline,candidate)
    assert report['comparable'] and report['survival_is_right_censored']
    assert report['survival_wins']==1 and report['score_wins']==1
    assert [p['seed'] for p in report['per_seed']]==[1,2]


def test_caps_and_different_evaluator_versions_are_not_compared():
    for key,value in [('max_pieces',10000),('evaluation_source_sha256',{'evaluate.py':'changed'})]:
        candidate=result();candidate[key]=value
        assert not module.compare(result(),candidate)['comparable']


def test_duplicate_rows_or_missing_seeds_are_rejected():
    candidate=result();candidate['per_seed'][1]['seed']=1
    assert not module.compare(result(),candidate)['comparable']
    candidate=result();candidate['seeds']=[1,3];candidate['per_seed'][1]['seed']=3
    assert not module.compare(result(),candidate)['comparable']


def test_large_development_regression_requests_review_without_promoting():
    candidate=result()
    for row in candidate['per_seed']:row['score']//=2
    report=module.compare(result(),candidate)
    assert report['review_recommended']
    assert report['mean_observed_score_ratio']==0.5
    assert 'no promotion or automatic stop' in report['interpretation']


def test_legacy_results_without_model_identity_are_not_silently_reused():
    baseline=result();baseline.pop('model_sha256')
    assert not module.compare(baseline,result())['comparable']
