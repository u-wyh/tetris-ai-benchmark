"""Calibrate natural-greedy diagnostics against independently stored A/D trajectories."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from own_policy_capture import sha,read_chain
from policy_contexts import prepare_run
from run_own_death_factorial import read_group,source_traces
from verify_feature_backend_benchmark import identical
from training.evaluation.evaluate_resumable import durable_json
from resource_guard import now


def restart_job(job):
    if any(job.glob('chunk_[0-9][0-9][0-9][0-9]')) or (job/'restart_receipt.json').exists():
        raise RuntimeError('Calibration outputs already exist; inspect/continue rather than repeat initial verifier')
    def execute():
        command=[sys.executable,str(ROOT/'scripts/evaluate_policy_contexts.py'),'--run-dir',str(job),'--max-new-chunks','1']
        child=subprocess.Popen(command)
        if child.wait():raise RuntimeError('Diagnostic evaluation child failed: '+str(child.pid))
        return child.pid,json.loads((job/f'process_{child.pid}.json').read_text())
    first,first_process=execute()
    if first_process['computed_chunks']!=[1] or first_process['reused_committed_chunks']!=0:
        raise ValueError('Calibration first process did not commit exactly the initial bounded prefix')
    before={str(path):sha(path) for directory in sorted(job.glob('chunk_[0-9][0-9][0-9][0-9]'))
            for path in directory.iterdir() if path.is_file()}
    second,second_process=execute()
    if (first==second or second_process['reused_committed_chunks']!=2 or second_process['computed_chunks']!=[2] or
            any(sha(path)!=expected for path,expected in before.items())):
        raise ValueError('Calibration process boundary changed a committed prefix')
    durable_json(job/'restart_receipt.json',dict(at=now(),status='passed',first_pid=first,second_pid=second,
        first_prefix_files_unchanged=True,protected_files=before,first_process=first_process,second_process=second_process))


def job_rows(job):
    _,_,identity,snapshot,last_hash,count=prepare_run(job)
    rows=[[] for _ in range(8)]
    for number in range(1,count):
        with np.load(job/f'chunk_{number:04d}/rows.npz',allow_pickle=False) as archive:
            data={key:archive[key] for key in archive.files}
        for row in range(len(data['lane'])):
            values={key:(value[row].copy() if value.ndim>1 else value[row].item()) for key,value in data.items()}
            rows[int(data['lane'][row])].append(values)
    summary=json.loads((job/'evaluation_summary.json').read_text())
    if (summary['status']!='completed' or summary['identity']!=identity or
            summary['last_manifest_sha256']!=last_hash or summary['actual_environment_steps']!=sum(map(len,rows))):
        raise ValueError('Completed diagnostic cache and work accounting differ')
    return rows,snapshot,summary


def run(folder):
    design=json.loads((folder/'design.json').read_text())
    def unchanged():
        for path,expected in design['frozen_files'].items():
            if sha(path)!=expected:raise ValueError('Frozen calibration input changed: '+path)
    unchanged()
    if (folder/'calibration_summary.json').exists():raise RuntimeError('Calibration already recorded; inspect, do not repeat')
    jobs={name:Path(path) for name,path in design['jobs'].items()}
    if set(jobs)!={'reference','candidate'}:raise ValueError('Both predeclared checkpoints required')
    if any(any(job.glob('chunk_[0-9][0-9][0-9][0-9]')) for job in jobs.values()):
        raise RuntimeError('Partial calibration exists; preserve its progress and inspect direct continuation')
    golden={}
    for group in range(4):
        directory=Path(design['golden_factorial_dir'])/f'group_{group:02d}'
        if sha(directory/'manifest.json')!=design['frozen_files'][str(directory/'manifest.json')]:
            raise ValueError('Frozen golden group manifest differs')
        identity=json.loads((directory/'manifest.json').read_text())['identity']
        data,states=read_group(directory,identity)
        for case in data['cases']:
            golden[case['target']['key']]=dict(target=case['target'],arms=case['arms'],states=states[case['target']['key']])
    reports={};total_steps=total_forward=0
    for name,job in jobs.items():
        restart_job(job);unchanged()
        rows,snapshot,summary=job_rows(job)
        suite=json.loads(Path(summary['suite']['path']).read_text())
        keys=[entry['key'] for entry in suite['contexts']]
        if len(golden)!=8 or set(keys)!=set(golden):raise ValueError('Calibration must use the exact frozen eight cases')
        originals=None
        if name=='candidate':
            selection=json.loads(Path(design['targets_path']).read_text())
            originals=source_traces(Path(selection['capture_dir']),[golden[key]['target'] for key in keys],
                                   selection['capture_identity'],design['frozen_files'])
        arm='A' if name=='reference' else 'D';checks=[]
        for lane,key in enumerate(keys):
            expected=golden[key]['arms'][arm];observed=rows[lane]
            exact=(len(observed)==len(expected['trace']) and
                   all(all(row[field]==value for field,value in old.items()) for row,old in zip(observed,expected['trace'])))
            endpoint=summary['endpoints'][lane];old=expected['summary']
            metrics=(endpoint['steps']==old['steps'] and endpoint['cleared_lines']==old['cleared_lines'] and
                     endpoint['score']==old['score_delta'] and abs(endpoint['reward']-old['reward'])<=1e-8 and
                     all(endpoint[field]==old[field] for field in ['game_over','time_limit','horizon_censored']))
            state=identical(snapshot['workers'][lane]['state'],golden[key]['states'][arm])
            previews=None
            if originals is not None:
                previews=(len(observed)==len(originals[lane]) and all(
                    row['observation_sha256']==source['digest']['observation'] and
                    row['mask_sha256']==source['digest']['mask'] and
                    np.array_equal(row['selected_features'],source['features'])
                    for row,source in zip(observed,originals[lane])))
            checks.append(dict(key=key,first_natural_action=observed[0]['action'],
                expected_first_action=expected['trace'][0]['action'],steps=len(observed),
                all_trace_fields_exact=exact,complete_endpoint_state_exact=state,endpoint_metrics_match=metrics,
                original_source_previews_checked=originals is not None,original_source_previews_exact=previews))
        passed=all(check['all_trace_fields_exact'] and check['complete_endpoint_state_exact'] and
            check['endpoint_metrics_match'] and check['original_source_previews_exact'] is not False for check in checks)
        reports[name]=dict(calibrated=passed,checks=checks,actual_environment_steps=summary['actual_environment_steps'],
            actual_policy_forward_calls=summary['actual_policy_forward_calls'],model_audit=summary['model_audit'],
            deaths=sum(endpoint['game_over'] for endpoint in summary['endpoints']),
            horizon_censored=sum(endpoint['horizon_censored'] for endpoint in summary['endpoints']))
        total_steps+=summary['actual_environment_steps'];total_forward+=summary['actual_policy_forward_calls']
    unchanged();passed=all(report['calibrated'] for report in reports.values())
    durable_json(folder/'calibration_summary.json',dict(at=now(),status='passed' if passed else 'failed',
        eligible_for_context_diagnostics=passed,jobs=reports,actual_new_environment_steps=total_steps,
        actual_policy_forward_calls=total_forward,optimizer_steps=0,policy_promotion='none',
        interpretation='Natural first action, new8-lane batching, real checkpoint reload and preserved prefixes. '
            'Reference matches frozenA; rejected8k matches sourceD. Eight correlated development contexts only; '
            'calibration is not a full quality gate or independent-seed validation.'))
    if not passed:raise RuntimeError('Natural diagnostic calibration differed from frozen trajectories; inspect scope, do not force actions')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    run(parser.parse_args().run_dir.resolve())
