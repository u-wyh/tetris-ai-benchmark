import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCRIPT=Path(__file__).resolve().parents[1]/'scripts/guarded_process.py'


def heartbeat(path,pid,age=0):
    path.write_text(json.dumps(dict(pid=pid,monotonic=time.monotonic()-age)))


def test_healthy_watchdog_preserves_successful_exit(tmp_path):
    heart=tmp_path/'heartbeat.json';heartbeat(heart,os.getpid())
    result=subprocess.run([sys.executable,str(SCRIPT),'--guardian-pid',str(os.getpid()),
                           '--heartbeat',str(heart),'--',sys.executable,'-c','pass'],timeout=5)
    assert result.returncode==0
    assert json.loads((tmp_path/'watchdog_result.json').read_text())['reason']=='command_completed'


def test_stale_heartbeat_prevents_starting_work(tmp_path):
    heart=tmp_path/'heartbeat.json';heartbeat(heart,os.getpid(),age=60)
    marker=tmp_path/'should_not_exist'
    result=subprocess.run([sys.executable,str(SCRIPT),'--guardian-pid',str(os.getpid()),
                           '--heartbeat',str(heart),'--',sys.executable,'-c',
                           f'from pathlib import Path; Path({str(marker)!r}).touch()'],timeout=5)
    assert result.returncode!=0
    assert not marker.exists()


def test_killed_guardian_stops_running_child(tmp_path):
    guardian=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    heart=tmp_path/'heartbeat.json';heartbeat(heart,guardian.pid)
    marker=tmp_path/'child_pid'
    wrapper=subprocess.Popen([sys.executable,str(SCRIPT),'--guardian-pid',str(guardian.pid),
                              '--heartbeat',str(heart),'--',sys.executable,'-c',
                              f'import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)'])
    try:
        deadline=time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:
            time.sleep(0.05)
        assert marker.exists()
        child=int(marker.read_text())
        guardian.kill();guardian.wait(timeout=5)
        assert wrapper.wait(timeout=5)!=0
        assert not Path(f'/proc/{child}').exists()
        assert json.loads((tmp_path/'watchdog_result.json').read_text())['reason']=='guardian_exited'
    finally:
        for process in (guardian,wrapper):
            if process.poll() is None:
                process.kill();process.wait(timeout=5)
