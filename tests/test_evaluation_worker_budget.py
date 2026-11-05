import pytest

from training.evaluation.worker_budget import automatic_workers,detect_resources,MIB


def resources(**changes):
    return dict(cpu_count=16,affinity_count=6,cgroup_cpu_quotas=[2.5],
                memory_headroom_bytes=2200*MIB,reserved_memory_bytes=512*MIB,
                worker_memory_budget_bytes=512*MIB,**changes)


def test_training_quota_limits_auto_evaluation_to_two_workers():
    assert automatic_workers(8,resources())==2


def test_parent_quota_and_affinity_both_apply():
    r=resources();r['cgroup_cpu_quotas']=[8,1.5]
    assert automatic_workers(8,r)==1
    r.update(cgroup_cpu_quotas=[],affinity_count=2)
    assert automatic_workers(8,r)==2


def test_small_memory_headroom_reduces_workers_or_refuses_start():
    r=resources();r.update(cgroup_cpu_quotas=[],memory_headroom_bytes=1200*MIB)
    assert automatic_workers(8,r)==1
    r['memory_headroom_bytes']=900*MIB
    with pytest.raises(RuntimeError,match='Insufficient memory'):
        automatic_workers(8,r)


def test_no_quota_does_not_exceed_seeds_or_eight_workers():
    r=resources();r.update(cgroup_cpu_quotas=[],affinity_count=16,memory_headroom_bytes=12000*MIB)
    assert automatic_workers(3,r)==3
    assert automatic_workers(20,r)==8


def test_nested_cgroup_limits_and_actual_usage_are_detected(tmp_path,monkeypatch):
    root=tmp_path/'cgroup';leaf=root/'team/job';leaf.mkdir(parents=True)
    (root/'team/cpu.max').write_text('100000 100000')
    (leaf/'cpu.max').write_text('50000 20000')
    (leaf/'memory.max').write_text(str(6*1024*MIB))
    (leaf/'memory.current').write_text(str(4*1024*MIB))
    (root/'team/memory.max').write_text(str(8*1024*MIB))
    (root/'team/memory.current').write_text(str(7*1024*MIB))
    cgroup=tmp_path/'self_cgroup';cgroup.write_text('0::/team/job\n')
    meminfo=tmp_path/'meminfo';meminfo.write_text('MemAvailable: 8388608 kB\n')
    monkeypatch.setattr('os.cpu_count',lambda:16)
    monkeypatch.setattr('os.sched_getaffinity',lambda _:set(range(6)))
    budget=detect_resources(root,cgroup,meminfo)
    assert budget['cgroup_cpu_quotas']==[2.5,1.0]
    assert budget['memory_headroom_bytes']==1024*MIB
    assert automatic_workers(8,budget)==1
