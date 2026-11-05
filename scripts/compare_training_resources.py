"""Select a hardware execution profile using screened and sustained PPO trials."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from resource_guard import ROOT, now, operation, read_telemetry, write_json


def run(output):
    sys.path.insert(0,str(ROOT))
    from training.evaluation.hybrid_queue import PPO,digest,verify_models
    from training.train_vector_transaction import verify_task
    verify_models()
    output.mkdir(parents=True,exist_ok=False)
    base_profile=json.loads((ROOT/'configs/training_resources.json').read_text())
    candidates=[('cpu8_q250','cpu',8,250),('cuda4_q250','cuda',4,250),
                ('cuda8_q200','cuda',8,200),('cuda8_q250','cuda',8,250),
                ('cuda8_q300','cuda',8,300),('cuda16_q250','cuda',16,250)]
    protocol=dict(at=now(),purpose='Hardware throughput selection, not model-quality evaluation',
                  model=str(PPO),model_sha256=digest(PPO),seed=1493000,
                  screening_steps=12288,sustained_steps=65536,
                  candidates=candidates,memory_headroom_fraction=0.15,
                  selection='Highest last-four-transaction throughput among safe sustained finalists',
                  guard_profile=base_profile)
    write_json(output/'protocol.json',protocol)
    operation('开始六种硬件执行配置筛选，每组12288步，前两名各持续65536步；不启动长期训练',output)
    screened=[]
    sustained=[]

    def trial(candidate,tasks,phase):
        label,device,workers,quota=candidate
        path=output/f'{phase}_{label}'
        path.mkdir()
        profile=dict(base_profile,initial_cpu_quota=quota,max_cpu_quota=quota)
        write_json(path/'resource_profile.json',profile)
        config=dict(target_total_steps=tasks*4096,task_steps=4096,device=device,
                    n_envs=workers,n_steps=4096//workers,base_seed=1493000,
                    candidate_policy=True,fast_candidate_env=True,max_pieces=100000,
                    overnight_monitor=True,step0_baseline=False,transaction_retention=max(3,tasks),
                    validation_interval=100000000,milestone_interval=100000000,
                    final_test_seeds=100,final_test_max_pieces=50000,
                    initial_model=dict(path=str(PPO),sha256=digest(PPO)),
                    ppo_overrides=dict(gamma=0.999,gae_lambda=0.98))
        write_json(path/'preset.json',config)
        write_json(output/'status.json',dict(at=now(),phase=phase,candidate=label))
        cooldown=time.monotonic()+180
        while True:
            s=read_telemetry(True)
            if s['cpu_c']<60 and s['gpu_c']<50:
                break
            if time.monotonic()>=cooldown:
                raise RuntimeError('Cooldown budget exhausted')
            time.sleep(5)
        cmd=[sys.executable,str(ROOT/'scripts/resource_guard.py'),'--profile',str(path/'resource_profile.json'),
             '--output',str(path/'guard'),'--seconds','420','--',sys.executable,'-u',
             '-m','training.train_vector_transaction','--run-dir',str(path/'training'),
             '--config-file',str(path/'preset.json'),'--max-tasks',str(tasks)]
        completed=subprocess.run(cmd,cwd=ROOT)
        guard=json.loads((path/'guard/result.json').read_text())
        samples=[json.loads(x) for x in (path/'guard/samples.jsonl').read_text().splitlines()]
        late=[s for s in samples if s['elapsed_seconds']>=samples[-1]['elapsed_seconds']-60]
        result=dict(candidate=label,phase=phase,device=device,n_envs=workers,quota_ceiling=quota,
                    guard_result=guard,cpu_peak_c=max(s['cpu_c'] for s in samples),
                    gpu_peak_c=max(s['gpu_c'] for s in samples),
                    late_cpu_peak_c=max(s['cpu_c'] for s in late),
                    late_gpu_peak_c=max(s['gpu_c'] for s in late),
                    peak_memory_mib=max(s['memory_mib'] for s in samples),
                    gpu_peak_memory_mib=max(s['gpu_memory_mib'] for s in samples),
                    gpu_mean_power_w=statistics.mean(s['gpu_power_w'] for s in samples if s['gpu_power_w'] is not None),
                    final_quota=guard['final_cpu_quota'],result_path=str(path/'result.json'))
        if completed.returncode==0 and guard['outcome']=='completed':
            metrics=[]
            frozen=json.loads((path/'training/config.json').read_text())
            for i in range(1,tasks+1):
                task=path/'training/committed'/f'task_{i:06d}'
                verify_task(task,i,frozen)
                metrics.append(json.loads((task/'metrics.json').read_text()))
            result.update(committed_steps=tasks*4096,transactions_verified=tasks,
                          steps_per_second=tasks*4096/sum(m['wall_seconds'] for m in metrics),
                          steady_steps_per_second=len(metrics[-4:])*4096/sum(m['wall_seconds'] for m in metrics[-4:]))
            result['eligible']=(result['peak_memory_mib']<=profile['memory_max_gib']*1024*0.85
                                and result['late_cpu_peak_c']<profile['cpu_soft_c']
                                and result['late_gpu_peak_c']<profile['gpu_soft_c'])
            result['reason']='within sustained safety/headroom policy' if result['eligible'] else 'insufficient thermal or memory headroom'
        else:
            result.update(eligible=False,reason=guard['outcome'])
        write_json(path/'result.json',result)
        print(json.dumps(result),flush=True)
        return result

    try:
        for candidate in candidates:
            screened.append(trial(candidate,3,'screen'))
            write_json(output/'screening.json',screened)
        eligible=sorted((r for r in screened if r['eligible']),key=lambda r:r['steady_steps_per_second'],reverse=True)
        if not eligible:
            raise RuntimeError('No safe configuration qualified')
        for r in eligible[:2]:
            c=next(c for c in candidates if c[0]==r['candidate'])
            sustained.append(trial(c,16,'sustained'))
            write_json(output/'sustained.json',sustained)
        safe=[r for r in sustained if r['eligible']]
        if not safe:
            raise RuntimeError('No sustained finalist qualified')
        winner=max(safe,key=lambda r:r['steady_steps_per_second'])
        summary=dict(at=now(),screening=screened,sustained=sustained,winner=winner,
                     model_unchanged=digest(PPO)==protocol['model_sha256'])
        write_json(output/'summary.json',summary)
        write_json(output/'status.json',dict(at=now(),phase='completed',winner=winner['candidate']))
        operation(f"六组筛选与持续对照完成，候选最优{winner['candidate']}，稳定段{winner['steady_steps_per_second']:.1f}步/秒；尚未启动长期训练",output/'summary.json')
    except BaseException as exc:
        write_json(output/'status.json',dict(at=now(),phase='failed',error=repr(exc)))
        operation(f'配置对照中止：{exc!r}，保留已完成数据',output)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.output.resolve())
