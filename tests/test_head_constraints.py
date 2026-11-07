import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import solve_head_constraints as lp
from test_head_representation import fixture


@pytest.fixture(scope='module')
def solver():
    name=os.environ.get('TETRIS_HEAD_SOLVER_MANIFEST')
    if not name:pytest.skip('Sealed local numerical solver artifact not configured')
    return lp.load_solver(Path(name),os.environ['TETRIS_HEAD_SOLVER_SHA256'])[0]


@pytest.mark.parametrize('sign',[1.,-1.])
def test_single_lp_finds_known_minimum_l1_with_both_displacement_signs(solver,sign):
    result=lp.solve_once([[sign,0.],[0.,1.]],[2.,3.],solver)
    assert result['classification']=='optimal' and result['solver_calls']==1
    assert np.allclose(result['delta'],[sign*2.,3.],atol=1e-9)
    assert result['l1_displacement']==pytest.approx(5.)
    assert result['max_constraint_violation']<=1e-7


def test_conflicting_joint_constraints_are_not_reported_as_a_fitted_policy(solver):
    result=lp.solve_once([[1.],[-1.]],[1.,1.],solver)
    assert result['classification']=='infeasible_reported' and 'delta' not in result
    assert result['solver_calls']==1


def test_time_limit_or_ambiguous_status_is_not_infeasibility(solver):
    assert lp.classify_status(solver.HighsModelStatus.kTimeLimit,solver)=='unresolved'
    assert lp.classify_status(solver.HighsModelStatus.kUnboundedOrInfeasible,solver)=='unresolved'


def test_validation_features_and_duplicate_lane_never_enter_training_constraints(tmp_path):
    arrays,metadata=fixture(tmp_path)
    matrix,rhs,origin=lp.constraints(arrays,metadata)
    assert len(origin)==320 and sum(p['kind']=='preference' for p in origin)==4
    for i,state in enumerate(metadata['states']):
        if state['split']=='validation' or (i<8 and i%2):
            arrays['h'][arrays['offsets'][i]:arrays['offsets'][i+1]]=1000.
    other,other_rhs,other_origin=lp.constraints(arrays,metadata)
    assert np.array_equal(matrix,other) and np.array_equal(rhs,other_rhs) and origin==other_origin


def test_pair_outranking_does_not_claim_global_top1_or_gpu_inference(tmp_path):
    arrays,metadata=fixture(tmp_path)
    arrays.update(h=np.zeros((1920,64),np.float32),actions=np.tile(np.array([0,1,2],np.int64),640),
                  offsets=np.arange(0,1923,3,dtype=np.int64),raw_score=np.zeros(1920,np.float32))
    for i in range(8):
        start=arrays['offsets'][i];arrays['h'][start+1,0]=1.;arrays['h'][start+2,0]=2.
    delta=np.r_[1.,np.zeros(63)]
    result=lp.summarize_solution(arrays,metadata,delta)
    assert all(p['new_margin']>0 and p['preferred_rank']==2 and not p['preferred_is_global_top1'] for p in result['preferences'])
    assert result['normal_summary']['validation']['proposed_double_changes']==0
    assert 'No updated GPU policy' in result['scope']


def test_solver_file_checksum_failure_precedes_any_import_or_model_call(tmp_path):
    file=tmp_path/'binary';file.write_text('changed')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(files={str(file):'wrong'},artifact_root=str(tmp_path))))
    with pytest.raises(ValueError,match='Frozen'):lp.load_solver(manifest,lp.sha(manifest))
