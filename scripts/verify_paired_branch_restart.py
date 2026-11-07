"""Compare complete sealed paired capture state/trace across a real process restart."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from own_policy_capture import sha
from restore_own_death_contexts import load_verified_chunk
from verify_feature_backend_benchmark import identical
from training.evaluation.evaluate_resumable import durable_json
from resource_guard import now


def run(folder):
    design=json.loads((folder/'restart_design.json').read_text())
    if (folder/'restart_result.json').exists():
        raise FileExistsError('Restart comparison already closed')
    for path,want in design['frozen_files'].items():
        if sha(path)!=want:raise ValueError('Frozen restart verification input changed')
    jobs=[Path(design[k]) for k in ('continuous','resumed')]
    locks=[];chains=[]
    try:
        numeric=[]
        for job in jobs:
            lock=(job/'capture.lock').open('r');fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB);locks.append(lock)
            jd=json.loads((job/'design.json').read_text())
            numeric.append({k:jd[k] for k in ['version','model','cases','horizon','chunk_rounds',
                'training_sources','inference_sources','evaluation_runtime','native_feature_runtime','legacy_schema']})
            identity=json.loads((job/'chunk_0000/manifest.json').read_text())['identity']
            chunks=[]
            for i in range(3):
                path=job/f'chunk_{i:04d}'
                chunks.append(load_verified_chunk(path,design['frozen_files'][str(path/'manifest.json')],identity))
            if [saved['round'] for saved,_,_ in chunks]!=[0,256,512]:
                raise ValueError('Real restart comparison must cover exactly0/256/512rounds')
            chains.append(chunks)
        if numeric[0]!=numeric[1]:raise ValueError('Continuous/resumed numerical definitions differ')
        for left,right in zip(*chains):
            if not identical(left[0],right[0]) or not identical(left[1],right[1]):
                raise ValueError('Complete worker/core/NumPy/action RNG/view or saved trace differs')
            for manifest in (left[2],right[2]):
                if manifest['audit'].get('policy_adam_counts_rng_unchanged') is not True:
                    raise ValueError('Source state audit missing')
        receipts=[json.loads(Path(path).read_text()) for path in design['phase_receipts']]
        if (len(receipts)!=3 or any(r['status']!='completed' for r in receipts) or
                len({r['pid'] for r in receipts})!=3 or
                [r['work']['gpu_forward_calls'] for r in receipts]!=[512,256,256] or
                receipts[0]['work']['environment_steps'] != sum(r['work']['environment_steps'] for r in receipts[1:])):
            raise ValueError('Not a real distinct-process256+256 versus512 comparison')
        for path,want in design['frozen_files'].items():
            if sha(path)!=want:raise ValueError('Restart input changed during comparison')
        durable_json(folder/'restart_result.json',dict(at=now(),status='passed',
            complete_worker_core_numpy_action_rng_views_equal=True,all_saved_trace_arrays_equal=True,
            compared_rounds=[0,256,512],unique_environment_steps_per_job=chains[0][-1][0]['environment_steps'],
            source_process_pids=[r['pid'] for r in receipts],
            source_environment_steps=sum(r['work']['environment_steps'] for r in receipts),
            source_gpu_forward_calls=sum(r['work']['gpu_forward_calls'] for r in receipts),
            optimizer_steps=0,new_environment_steps=0,gpu_forward_calls=0,
            eligible_for_remaining_bounded_branch_phases=True,eligible_for_training=False,
            scope='Fixed4 previously positive pairs, original frozen CUDA30M, paired duplicate seeds with distinctbranchIDs. Restart equivalence only, not5000-window labels or historical runtime reproduction.'))
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    run(parser.parse_args().run_dir.resolve())
