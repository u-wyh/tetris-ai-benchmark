"""Freeze one explicit paired checkpoint boundary; this command never launches compute."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from paired_branch_capture import stored_boundary,sha,VERSION
from paired_branch_guard import require_authorization
from training.evaluation.evaluate_resumable import durable_json


def prepare(job,attempt,mode,chunks):
    design=json.loads((job/'design.json').read_text())
    if (design.get('version')!=VERSION or design['job_folder']!=str(job) or
            mode not in ('capture','probe') or type(chunks) is not int or not 1<=chunks<=2):
        raise ValueError('Explicit paired job, mode or phase size differs')
    require_authorization(design,attempt)
    if attempt.exists():raise FileExistsError('Attempt exists; inspect its actual receipt before retry')
    lock=(job/'capture.lock').open('a');fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
    try:
        for path,want in design['frozen_files'].items():
            if sha(path)!=want:raise ValueError('Frozen paired source/input changed: '+path)
        before=stored_boundary(job)
        attempt.mkdir(parents=True)
        profile=ROOT/'configs/training_resources.json'
        (attempt/'profile.json').write_bytes(profile.read_bytes())
        request=dict(version='paired-phase-v1',mode=mode,job_folder=str(job),job_design_sha256=sha(job/'design.json'),
            expected_start=before,max_chunks=chunks,controller_design=str(attempt/'design.json'),guard_dir=str(attempt/'guard'))
        durable_json(attempt/'request.json',request)
        frozen={**design['frozen_files'],str(job/'design.json'):sha(job/'design.json')}
        # Every existing state/row/manifest is externally sealed before any later pickle.
        for directory in sorted(job.glob('chunk_[0-9][0-9][0-9][0-9]')):
            for name in ('manifest.json','state.pkl','rows.npz'):
                frozen[str(directory/name)]=sha(directory/name)
        for path in (attempt/'request.json',attempt/'profile.json',Path(__file__).resolve()):
            frozen[str(path)]=sha(path)
        durable_json(attempt/'design.json',dict(thread_id=design['thread_id'],control=design['control'],
            upstream_units=[],command_seconds=120,wall_seconds=600,frozen_files=frozen,
            validation_command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/run_paired_branch_phase.py'),
                                '--attempt-dir',str(attempt)],
            description='原30M配对GPU有界阶段完成；实际工作量见source receipt，完整策略/Adam/计数/RNG保护保持',
            activation='Explicit paired phase only; at most2x256rounds/4096newsteps, no optimizer or model promotion'))
        if stored_boundary(job)!=before:raise ValueError('Job changed during explicit phase preparation')
        print(json.dumps(dict(attempt=str(attempt),expected_start=before,max_chunks=chunks,mode=mode)))
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-folder',required=True,type=Path);p.add_argument('--attempt-dir',required=True,type=Path)
    p.add_argument('--mode',choices=['capture','probe'],default='capture')
    p.add_argument('--max-chunks',type=int,choices=[1,2],default=2)
    args=p.parse_args();prepare(args.job_folder.resolve(),args.attempt_dir.resolve(),args.mode,args.max_chunks)
