"""Read-only cache status; a successful bounded command is not a complete evaluation.

No models/workers/forward/optimizer are started. Original jobs are never written.
The frozen capture preflight and terminal auditor run on temporary symlink views;
the completion marker is exposed only to the auditor, never used as proof itself.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import pickle
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audit_own_policy_capture as terminal_audit
from capture_gpu_evaluation import prepare, validate_snapshot
from own_policy_capture import sha
from restore_own_death_contexts import load_verified_chunk
from resource_guard import now


class InspectionBlocked(RuntimeError):
    pass


def input_paths(folder):
    paths = [folder/'design.json']
    for chunk in sorted(folder.glob('chunk_[0-9][0-9][0-9][0-9]')):
        if not chunk.is_dir():
            raise ValueError('Committed chunk is not a directory')
        paths.extend(chunk/name for name in ['manifest.json', 'state.pkl', 'rows.npz'])
    paths.extend(sorted(folder.glob('process_*.json')))
    if (folder/'capture_summary.json').exists():
        paths.append(folder/'capture_summary.json')
    return paths


def fingerprint(folder):
    return {str(p): sha(p) for p in input_paths(folder)}


@contextmanager
def stable_job(folder):
    """Share an existing lock without creating it; refuse a live writer or a race."""
    path = folder/'capture.lock'
    lock = None
    try:
        try:
            lock = path.open('rb')
        except FileNotFoundError:
            pass
        if lock is not None:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise InspectionBlocked('Capture writer holds the lock; inspect again after its commit') from exc
            inode = lock.fileno()
            import os
            stamp = (os.fstat(inode).st_dev, os.fstat(inode).st_ino)
        else:
            stamp = None
        before = fingerprint(folder)
        yield before
        after = fingerprint(folder)
        current = path.stat() if path.exists() else None
        if (before != after or (stamp is None and current is not None) or
                (stamp is not None and (current is None or stamp != (current.st_dev, current.st_ino)))):
            raise InspectionBlocked('Capture inputs or lock changed during inspection; retry without starting work')
    finally:
        if lock is not None:
            lock.close()


def inspect(folder, upstream_summary=None):
    folder = Path(folder).resolve()
    result = dict(at=now(), folder=str(folder), completed=False, can_resume=False,
        status_version='gpu8-status-v1', configured_lanes=8,
        inspector_sources={str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
                           str(ROOT/'scripts/audit_own_policy_capture.py'): sha(ROOT/'scripts/audit_own_policy_capture.py')},
        new_environment_steps=0, gpu_forward_calls=0, optimizer_steps=0,
        scope='Cache status only; no scheduling, quality qualification, policy promotion or GPU readiness claim')
    try:
        with stable_job(folder) as protected, tempfile.TemporaryDirectory(prefix='tetris-evaluation-status-') as tmp:
            view = Path(tmp)
            # The preflight refuses summary markers because it is also used by
            # the compute entry. This read-only view retains ALL protocol checks
            # without invoking that compute entry or changing the original marker.
            (view/'design.json').symlink_to(folder/'design.json')
            for chunk in sorted(folder.glob('chunk_[0-9][0-9][0-9][0-9]')):
                (view/chunk.name).symlink_to(chunk, target_is_directory=True)
            design, identity, (snapshot, last_sha, count) = prepare(view)
            round_limit = design.get('phase_round_limit', 20000)
            step_limit = design.get('phase_environment_step_limit', 160000)
            if (type(round_limit) is not int or not 1 <= round_limit <= 20000 or
                    type(step_limit) is not int or not 1 <= step_limit <= 160000):
                raise ValueError('Declared finite phase budget is invalid')
            result.update(identity=identity, committed_chunks=count, last_manifest_sha256=last_sha,
                round=0 if snapshot is None else snapshot['round'],
                actual_unique_environment_steps=0 if snapshot is None else snapshot['environment_steps'],
                active_lanes=[] if snapshot is None else
                    [i for i, worker in enumerate(snapshot['workers']) if worker['active']],
                uncommitted_paths=sorted(p.name for p in folder.glob('chunk_*.pending')),
                trace_endpoints_verified=False)
            phase_limit_reached = result['round'] >= round_limit or result['actual_unique_environment_steps'] >= step_limit
            result.update(phase_round_limit=round_limit, phase_environment_step_limit=step_limit,
                          phase_budget_reached=phase_limit_reached)
            # read_chain in prepare has already verified every manifest/file
            # before any pickle. Audit initial seed/reset invariants as well.
            if count:
                initial, _, _ = load_verified_chunk(view/'chunk_0000',
                    protected[str(folder/'chunk_0000/manifest.json')], identity)
                validate_snapshot(initial, design)
                for number in range(count):
                    manifest = json.loads((view/f'chunk_{number:04d}'/'manifest.json').read_text())
                    audit = manifest['audit']
                    if (audit.get('policy_adam_counts_unchanged') is not True or
                            audit.get('global_sampling_rng_unchanged') is not True or
                            audit.get('optimizer_steps') != 0 or
                            audit.get('model_counts') != {name: spec['counts'] for name, spec in design['models'].items()}):
                        raise ValueError('Committed model/counter/RNG audit differs')
            summary_path = folder/'capture_summary.json'
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if (snapshot is None or result['active_lanes'] or summary.get('status') != 'passed' or
                        summary.get('identity') != identity or len(summary.get('endpoints', [])) != 8 or
                        summary.get('final_round') != snapshot['round'] or
                        summary.get('environment_steps') != snapshot['environment_steps']):
                    raise ValueError('Completion marker contradicts committed state, identity or all eight endpoints')
                if upstream_summary is None:
                    raise InspectionBlocked('Stopped cache has a summary; protected controller receipt is required for terminal audit')
                upstream_summary = Path(upstream_summary).resolve()
                upstream_sha = sha(upstream_summary)
                if json.loads(upstream_summary.read_text()).get('status') != 'passed':
                    raise InspectionBlocked('Protected controller receipt has not passed; preserve stopped cache')
                (view/'capture_summary.json').symlink_to(summary_path)
                for process in folder.glob('process_*.json'):
                    (view/process.name).symlink_to(process)
                terminal_audit.run(view, upstream_summary)
                if sha(upstream_summary) != upstream_sha:
                    raise InspectionBlocked('Controller receipt changed during terminal audit')
                audit = json.loads((view/'trajectory_audit.json').read_text())
                if audit['status'] != 'passed' or len(audit['lanes']) != 8 or audit['identity'] != identity:
                    raise ValueError('Full terminal trace audit did not verify all eight lanes')
                result.update(status='terminal_verified', completed=True, trace_endpoints_verified=True,
                    terminal_audit=audit, controller_receipt_sha256=upstream_sha,
                    next_action='preserve_completed_job; no further environment work')
            elif snapshot is None:
                result.update(status='fresh', next_action='requires separate authorized CUDA8 resource-guarded launch')
            else:
                result.update(status='resumable_partial', can_resume=not phase_limit_reached or not result['active_lanes'],
                    next_action='finalize missing summary from committed stopped state; no new environment steps needed'
                        if not result['active_lanes'] else
                        ('declared finite phase budget reached; preserve result without automatic extension'
                         if phase_limit_reached else 'resume committed boundary under CUDA8 guards'))
            result['protected_input_files'] = protected
        return result
    except InspectionBlocked as exc:
        result.update(status='blocked', completed=False, can_resume=False, reason=str(exc))
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OSError, EOFError,
            RuntimeError, pickle.UnpicklingError, zipfile.BadZipFile) as exc:
        result.update(status='invalid', completed=False, can_resume=False, reason=str(exc))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--upstream-summary', type=Path)
    args = parser.parse_args()
    result = inspect(args.run_dir, args.upstream_summary)
    print(json.dumps(result))
    sys.exit({'invalid': 2, 'blocked': 3}.get(result['status'], 0))
