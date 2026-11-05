"""Choose automatic evaluation concurrency from the process's actual resources."""

import math
import os
from pathlib import Path, PurePosixPath

MIB = 1024**2


def detect_resources(cgroup_root=Path('/sys/fs/cgroup'),
                     cgroup_file=Path('/proc/self/cgroup'),
                     meminfo_file=Path('/proc/meminfo')):
    cpu_count=os.cpu_count() or 1
    affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else cpu_count
    quotas=[]
    memory=[]
    if meminfo_file.exists():
        for line in meminfo_file.read_text().splitlines():
            if line.startswith('MemAvailable:'):
                memory.append(int(line.split()[1])*1024)
    if cgroup_file.exists():
        group=next((line[3:] for line in cgroup_file.read_text().splitlines()
                    if line.startswith('0::')),None)
        if group is not None:
            if '..' in PurePosixPath(group).parts:
                raise ValueError('Invalid unified cgroup path')
            path=cgroup_root/group.lstrip('/')
            while True:
                cpu_max=path/'cpu.max'
                if cpu_max.exists():
                    quota,period=cpu_max.read_text().split()
                    if int(period)<=0:
                        raise ValueError('Invalid cgroup CPU period')
                    if quota!='max':
                        value=int(quota)/int(period)
                        if value<=0:
                            raise ValueError('Invalid cgroup CPU quota')
                        quotas.append(value)
                maximum=path/'memory.max'
                if maximum.exists() and maximum.read_text().strip()!='max':
                    # Include ancestor headroom because a parent can impose a tighter limit.
                    limit=int(maximum.read_text())
                    used=int((path/'memory.current').read_text())
                    memory.append(max(0,limit-used))
                if path==cgroup_root:
                    break
                path=path.parent
    return dict(cpu_count=cpu_count,affinity_count=affinity,cgroup_cpu_quotas=quotas,
                memory_headroom_bytes=min(memory) if memory else None,
                reserved_memory_bytes=512*MIB,worker_memory_budget_bytes=512*MIB)


def automatic_workers(seed_count, resources=None):
    if seed_count<1:
        raise ValueError('Evaluation needs at least one seed')
    resources=detect_resources() if resources is None else resources
    limits=[resources['cpu_count'],resources['affinity_count'],*resources['cgroup_cpu_quotas']]
    cpus=max(1,math.floor(min(limits)))
    workers=min(8,seed_count,cpus)
    headroom=resources['memory_headroom_bytes']
    if headroom is not None:
        memory_workers=(headroom-resources['reserved_memory_bytes'])//resources['worker_memory_budget_bytes']
        if memory_workers<1:
            raise RuntimeError('Insufficient memory headroom for one evaluation worker')
        workers=min(workers,memory_workers)
    return workers
