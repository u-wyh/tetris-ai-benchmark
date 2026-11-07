"""One fixed L1 score-head LP on sealed CUDA features; coefficients remain unapplied."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from freeze_head_representation import load_cache,sha,unchanged


def load_solver(manifest_path,expected_sha):
    if sha(manifest_path)!=expected_sha:raise ValueError('External solver manifest seal differs')
    manifest=json.loads(Path(manifest_path).read_text());unchanged(manifest['files'])
    artifact=Path(manifest['artifact_root']).resolve()
    if 'highspy' in sys.modules and artifact not in Path(sys.modules['highspy'].__file__).resolve().parents:
        raise ValueError('Unexpected preloaded solver outside the sealed artifact')
    sys.dont_write_bytecode=True;sys.path.insert(0,str(artifact))
    import highspy
    if artifact not in Path(highspy.__file__).resolve().parents:raise ValueError('Solver imported from another runtime')
    if highspy.Highs().version()!=manifest['version']:raise ValueError('Solver binary version differs')
    return highspy,manifest


def state_rows(arrays,index):
    start,stop=arrays['offsets'][index:index+2]
    return arrays['actions'][start:stop],arrays['h'][start:stop].astype(np.float64)


def constraints(arrays,metadata):
    w=arrays['original_weight'].astype(np.float64);deltas=[];targets=[];origin=[]
    for index,state in enumerate(metadata['states']):
        if state['kind']!='normal' or state['split']!='train':continue
        actions,h=state_rows(arrays,index);reference=int(np.flatnonzero(actions==state['reference_action'])[0])
        competitors=np.flatnonzero(actions!=state['reference_action'])
        difference=h[reference]-h[competitors];margin=difference@w
        deltas.append(difference);targets.extend(np.minimum(margin,1e-3)-margin)
        origin.extend(dict(kind='normal',state=index,action=int(actions[a])) for a in competitors)
    # Four verified pairs only; duplicate initial lanes never double their weight.
    for index in (0,2,4,6):
        state=metadata['states'][index];actions,h=state_rows(arrays,index)
        good=int(np.flatnonzero(actions==state['good_action'])[0]);base=int(np.flatnonzero(actions==state['reference_action'])[0])
        difference=h[good]-h[base]
        deltas.append(difference[None]);targets.append(1e-3-float(difference@w))
        origin.append(dict(kind='preference',state=index,action=state['good_action']))
    matrix=np.concatenate(deltas);rhs=np.asarray(targets,dtype=np.float64)
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():raise ValueError('Nonfinite fixed LP')
    return matrix,rhs,origin


def classify_status(status,highspy):
    if status==highspy.HighsModelStatus.kOptimal:return 'optimal'
    if status==highspy.HighsModelStatus.kInfeasible:return 'infeasible_reported'
    return 'unresolved'


def solve_once(matrix,rhs,highspy):
    """d=positive-negative: nonnegative128-variable LP equals minimumL1 displacement."""
    matrix=np.asarray(matrix,dtype=np.float64);rhs=np.asarray(rhs,dtype=np.float64)
    if (matrix.ndim!=2 or not len(matrix) or rhs.shape!=(len(matrix),) or
            not np.isfinite(matrix).all() or not np.isfinite(rhs).all()):raise ValueError('Invalid LP matrix/RHS')
    rows,dimensions=matrix.shape;columns=dimensions*2
    if rows*columns>=2**31:raise ValueError('LP exceeds fixed32bit sparse-index budget')
    solver=highspy.Highs();warnings=[]
    def checked(value):
        if value==highspy.HighsStatus.kError:raise RuntimeError('HiGHS API rejected the fixed problem/options')
        if value==highspy.HighsStatus.kWarning:warnings.append(str(value))
    options=dict(solver='simplex',threads=1,parallel='off',presolve='on',time_limit=60.,random_seed=0,
                 primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,small_matrix_value=1e-12,output_flag=False)
    for name,value in options.items():checked(solver.setOptionValue(name,value))
    lp=highspy.HighsLp();lp.num_col_=columns;lp.num_row_=rows;lp.sense_=highspy.ObjSense.kMinimize
    lp.col_cost_=np.ones(columns);lp.col_lower_=np.zeros(columns);lp.col_upper_=np.full(columns,highspy.kHighsInf)
    lp.row_lower_=rhs;lp.row_upper_=np.full(rows,highspy.kHighsInf)
    lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise;lp.a_matrix_.num_col_=columns;lp.a_matrix_.num_row_=rows
    lp.a_matrix_.start_=np.arange(0,(rows+1)*columns,columns,dtype=np.int32)
    lp.a_matrix_.index_=np.tile(np.arange(columns,dtype=np.int32),rows)
    lp.a_matrix_.value_=np.concatenate([matrix,-matrix],axis=1).reshape(-1)
    checked(solver.passModel(lp))
    checked(solver.run())  # Exactly one numerical LP solve; never a policy optimizer call.
    status=solver.getModelStatus();info=solver.getInfo();classification=classify_status(status,highspy)
    result=dict(classification=classification,model_status=str(status),solver_version=solver.version(),
        solver_calls=1,simplex_iterations=int(info.simplex_iteration_count),options=options,
        constraints=rows,displacement_dimensions=dimensions,nonnegative_variables=columns,api_warnings=warnings)
    if classification=='optimal':
        solution=solver.getSolution()
        if not solution.value_valid:raise ValueError('Optimal status lacks a valid primal solution')
        values=np.asarray(solution.col_value,dtype=np.float64);delta=values[:dimensions]-values[dimensions:]
        violation=max(0.,float(np.max(rhs-matrix@delta)))
        result.update(delta=delta.tolist(),l1_displacement=float(np.abs(delta).sum()),
            solver_objective=float(info.objective_function_value),max_constraint_violation=violation,
            numerical_residual_passed=violation<=1e-7)
        if violation>1e-7:result['classification']='numerical_validation_failed'
    return result


def summarize_solution(arrays,metadata,delta):
    w=arrays['original_weight'].astype(np.float64);new=w+np.asarray(delta)
    cast=new.astype(np.float32).astype(np.float64);changes=[];pairs=[];by_split={}
    for index,state in enumerate(metadata['states']):
        actions,h=state_rows(arrays,index);scores=h@new;old=h@w;cast_scores=h@cast
        predicted=int(actions[np.argmax(scores)]);cast_mode=int(actions[np.argmax(cast_scores)])
        old_mode=int(actions[np.argmax(old)])
        if state['kind']=='normal':
            if predicted!=state['reference_action'] or cast_mode!=state['reference_action'] or old_mode!=state['reference_action']:
                changes.append(dict(state=index,seed=state['seed'],chunk=state['chunk'],split=state['split'],
                    reference_action=state['reference_action'],original_double_mode=old_mode,
                    proposed_double_mode=predicted,proposed_float32_weight_double_eval_mode=cast_mode))
        elif index%2==0:
            good=int(np.flatnonzero(actions==state['good_action'])[0]);base=int(np.flatnonzero(actions==state['reference_action'])[0])
            pairs.append(dict(state=index,seed=state['seed'],reference_action=state['reference_action'],good_action=state['good_action'],
                original_margin=float(old[good]-old[base]),new_margin=float(scores[good]-scores[base]),
                float32_weight_margin=float(cast_scores[good]-cast_scores[base]),preferred_rank=1+int(np.sum(scores>scores[good])),
                predicted_global_action=predicted,preferred_is_global_top1=predicted==state['good_action']))
    for split in ('train','validation'):
        group=[s for s in metadata['states'] if s['kind']=='normal' and s['split']==split]
        by_split[split]=dict(states=len(group),episodes=len({s['seed'] for s in group}),
            proposed_double_changes=sum(p['split']==split and p['proposed_double_mode']!=p['reference_action'] for p in changes),
            cast_weight_changes=sum(p['split']==split and p['proposed_float32_weight_double_eval_mode']!=p['reference_action'] for p in changes),
            original_double_mismatches=sum(p['split']==split and p['original_double_mode']!=p['reference_action'] for p in changes))
    matrix,rhs,_=constraints(arrays,metadata)
    return dict(normal_summary=by_split,all_normal_discrepancies=changes,preferences=pairs,
        float32_weight_max_constraint_violation=max(0.,float(np.max(rhs-matrix@(cast-w)))),
        new_weight_max_abs=float(np.max(np.abs(new))),source_weight_l1=float(np.abs(w).sum()),
        scope='Float64 affine predictions on fixed CUDA h; cast weights still evaluated in float64. No updated GPU policy or whole-game quality evidence.')


def run(folder):
    import torch
    from paired_branch_guard import require_authorization
    from training.evaluation.evaluate_resumable import durable_json
    from resource_guard import now
    design=json.loads((folder/'lp_design.json').read_text());unchanged(design['frozen_files']);require_authorization(design,folder)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; preserve the sealed cache without host/CPU model fallback')
    if (folder/'execution.json').exists():raise FileExistsError('Single LP attempt exists; inspect rather than retry')
    receipt=dict(at=now(),status='started',lp_solver_calls=None,environment_steps=0,gpu_forward_calls=0,policy_optimizer_steps=0,model_loads=0)
    durable_json(folder/'execution.json',receipt)
    try:
        arrays,metadata=load_cache(Path(design['cache']),design['cache_metadata_sha256'],design['cache_arrays_sha256'])
        evidence=json.loads(Path(design['current_gpu_labels']).read_text())
        if evidence['label_counts']['beneficial']!=4 or evidence['status']!='passed':raise ValueError('Verified four-pair labels changed')
        for i,pair in enumerate(evidence['comparisons']):
            state=metadata['states'][i*2];case=pair['case']
            if (state['seed'],state['reference_action'],state['good_action'])!=(case['seed'],case['baseline_action'],case['action']):
                raise ValueError('Cached preference differs from current-GPU branch evidence')
        highspy,manifest=load_solver(design['solver_manifest'],design['solver_manifest_sha256'])
        matrix,rhs,origin=constraints(arrays,metadata)
        result=solve_once(matrix,rhs,highspy);receipt['lp_solver_calls']=1
        if result['classification']=='optimal':result['prediction_audit']=summarize_solution(arrays,metadata,result['delta'])
        unchanged(design['frozen_files'])
        result.update(at=now(),status='completed',constraint_rows_by_kind={kind:sum(p['kind']==kind for p in origin) for kind in ('normal','preference')},
            eligible_for_training=False,source_model_parameters_updated=False,candidate_coefficients_applied=False,
            work=dict(lp_solver_calls=1,environment_steps=0,gpu_forward_calls=0,policy_optimizer_steps=0,model_loads=0),
            scope='Single predetermined joint L1 head fit, coefficients only and unapplied.316train normal states/4training pairs;316validation normal states not fitted. Infeasible is solver-reported only for this fixed representation/constraint set; timeout/ambiguous statuses unresolved. No model ZIP, PPO training, promotion or quality gate change.')
        durable_json(folder/'single_lp_result.json',result)
        receipt.update(at=now(),status='completed',classification=result['classification'])
    except BaseException as exc:
        receipt.update(at=now(),status='failed',error=repr(exc));raise
    finally:durable_json(folder/'execution.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    run(p.parse_args().run_dir.resolve())
