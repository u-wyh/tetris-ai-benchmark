"""Hand off an eligible frozen parameter screen to extra validation and guarded training."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

from hourly_continue import inspect
from resource_guard import ROOT, now, operation, write_json
from compare_development_evaluations import compare
from evaluation_identity import module_for, runtime_identity, RESUMABLE


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_json(path, value):
    if path.exists():
        existing = json.loads(path.read_text())
        if {k:v for k,v in existing.items() if k != 'at'} != {k:v for k,v in value.items() if k != 'at'}:
            raise RuntimeError('Existing handoff differs; refusing overwrite: ' + str(path))
    else:
        write_json(path, value)


def run(folder):
    if (folder/'summary.json').exists():
        return
    p = json.loads((folder/'protocol.json').read_text())
    lock = (folder/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic()+p['wall_seconds']

    def interrupted(*args):
        raise KeyboardInterrupt('Followup interrupted')
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)

    def check():
        status,_ = inspect(Path.home()/'.codex', p['thread_id'])
        if status not in ('active','usageLimited','usage_limited') or (folder/'STOP').exists():
            raise RuntimeError('Goal stopped or STOP file set')
        if time.monotonic() >= deadline:
            raise RuntimeError('Followup wait budget exhausted')
        for name, sha in p['frozen_files'].items():
            if digest(name) != sha:
                raise RuntimeError('Frozen followup input changed: ' + name)

    try:
        extension_enabled = p.get('extension_enabled', True)
        if type(extension_enabled) is not bool:
            raise ValueError('extension_enabled must be a boolean')
        upstream = Path(p['parameter_screen'])
        write_json(folder/'status.json', dict(at=now(), stage='waiting_for_parameter_screen'))
        while not (upstream/'summary.json').exists():
            check()
            pid = subprocess.run(['systemctl','--user','show',p['upstream_service'],'-p','MainPID','--value'],
                                 capture_output=True,text=True,check=True,timeout=10).stdout.strip()
            if (upstream/'summary.json').exists():
                break
            state = json.loads((upstream/'status.json').read_text()) if (upstream/'status.json').exists() else {}
            bootstrap_live = False
            if pid == '0' and p.get('bootstrap_service') and state.get('stage') != 'stopped':
                bootstrap_pid = subprocess.run(['systemctl','--user','show',p['bootstrap_service'],
                    '-p','MainPID','--value'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
                bootstrap_live = bootstrap_pid.isdigit() and int(bootstrap_pid) > 0
            if state.get('stage') == 'stopped' or not pid.isdigit() or (int(pid) == 0 and not bootstrap_live):
                raise RuntimeError('Parameter screen stopped before producing a result')
            time.sleep(10)
        while subprocess.run(['systemctl','--user','is-active','--quiet',p['upstream_service']]).returncode == 0:
            check()
            time.sleep(3)
        check()
        if json.loads((upstream/'status.json').read_text())['stage'] != 'completed_screen':
            raise RuntimeError('Parameter screen has no successful terminal status')
        summary = json.loads((upstream/'summary.json').read_text())
        selected = summary['recommended_for_review']
        if selected is None:
            write_json(folder/'summary.json',dict(at=now(),outcome='no_eligible_candidate'))
            write_json(folder/'status.json',dict(at=now(),stage='no_eligible_candidate'))
            operation('参数比较没有合格候选，自动取消额外评测与续训',folder/'summary.json')
            return
        if not any(r['case'] == selected and r['eligible_for_longer_development'] for r in summary['results']):
            raise RuntimeError('Selection is inconsistent with eligibility evidence')
        source = json.loads((upstream/'protocol.json').read_text())
        case = upstream/selected
        baseline = json.loads(Path(source['baseline']).read_text())
        model = case/'stage1_checkpoint'/f'task_{source["stage1_steps"]//4096:06d}'/'model.zip'
        result = json.loads((case/source['evaluation_filename']).read_text())
        if (result['model_sha256'] != digest(model) or result['seeds'] != baseline['seeds']
                or result['max_pieces'] != 20000 or result['committed_steps'] != source['stage1_steps']):
            raise RuntimeError('Selected model differs from evaluated candidate')
        paired = compare(baseline, result)
        if (not paired['comparable'] or paired['mean_observed_score_ratio'] < 0.99
                or not all(row['survived_cap'] and not row['game_over'] for row in result['per_seed'])):
            raise RuntimeError('Selected candidate no longer passes the frozen screen')
        heldout = Path(p['heldout'])
        extension = Path(p['extension']) if extension_enabled else None
        heldout.mkdir(exist_ok=True)
        ensure_json(heldout/'seeds.json',dict(validation=p['extra_seeds']))
        resources = [(heldout,'resource_profile.json',p['evaluation_resources'])]
        if extension_enabled:
            extension.mkdir(exist_ok=True)
            resources += [(extension,'training_resources.json',p['training_resources']),
                          (extension,'evaluation_resources.json',p['evaluation_resources'])]
        for directory, name, origin in resources:
            ensure_json(directory/name,json.loads(Path(origin).read_text()))
        frozen = dict(p['frozen_files'])
        for path in [upstream/'summary.json',case/source['evaluation_filename'],model,
                     heldout/'seeds.json',heldout/'resource_profile.json']:
            frozen[str(path)] = digest(path)
        heldout_protocol = dict(at=now(),thread_id=p['thread_id'],wall_seconds=3600,diagnostic_required=False,
            evaluation_module=module_for(source),
            seeds=p['extra_seeds'],seeds_file=str(heldout/'seeds.json'),extension_gate=p['extension_gate'],
            training_source_sha256=source['training_source_sha256'],
            evaluation_source_sha256=baseline['evaluation_source_sha256'],frozen_files=frozen,
            models=dict(baseline=dict(path=baseline['model_checkpoint'],sha256=baseline['model_sha256'],committed_steps=0),
                        candidate=dict(path=str(model),sha256=result['model_sha256'],committed_steps=source['stage1_steps'])))
        if module_for(source) == RESUMABLE:
            heldout_protocol.update(runtime_identity())
        ensure_json(heldout/'protocol.json',heldout_protocol)
        if extension_enabled:
            extra_inputs = [heldout/'protocol.json',extension/'training_resources.json',extension/'evaluation_resources.json',
                        case/'preset.json',case/'training/config.json',Path(p['short_baseline'])]
            extension_protocol = dict(at=now(),thread_id=p['thread_id'],wall_seconds=21600,
            evaluation_module=module_for(source),
            longer_screen=str(heldout),upstream_service=p['heldout_service'],training=str(case/'training'),
            preset=str(case/'preset.json'),short_baseline=p['short_baseline'],seeds_file=str(heldout/'seeds.json'),
            checkpoints=[524288,1048576,1572864,2097152],extension_gate=p['extension_gate'],
            frozen_files={**frozen,**{str(path):digest(path) for path in extra_inputs}})
            ensure_json(extension/'protocol.json',extension_protocol)
        check()
        subprocess.run(['systemctl','--user','enable',p['extension_service'] if extension_enabled else p['heldout_service']],check=True,timeout=20)
        subprocess.run(['systemctl','--user','start',p['heldout_service']],check=True,timeout=20)
        if extension_enabled:
            subprocess.run(['systemctl','--user','start',p['extension_service']],check=True,timeout=20)
        write_json(folder/'summary.json',dict(at=now(),outcome='handed_off',selected_case=selected,
                   heldout=str(heldout),extension=str(extension) if extension else None,
                   extension_enabled=extension_enabled,model_promotion='none'))
        write_json(folder/'status.json',dict(at=now(),stage='handed_off',selected_case=selected))
        operation(f'参数比较自动接力：{selected}进入额外8种子长局复核；追加训练开关={extension_enabled}，无模型发布',folder/'summary.json')
    except BaseException as exc:
        write_json(folder/'status.json',dict(at=now(),stage='stopped',error=repr(exc)))
        operation(f'参数比较接力停止：{exc!r}，已保存进度',folder)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    run(parser.parse_args().run_dir.resolve())
