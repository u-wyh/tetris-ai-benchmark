"""Boot-time conditions must preserve pauses and closed or unknown goals."""

from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from development_allowed import allowed


@pytest.mark.parametrize('status,expected', [('active',True),('usageLimited',True),('usage_limited',True),
    ('paused',False),('budgetLimited',False),('blocked',False),('complete',False),('unknown',False)])
def test_startup_condition_respects_native_goal_status(tmp_path,status,expected):
    with sqlite3.connect(tmp_path/'goals_1.sqlite') as db:
        db.execute('CREATE TABLE thread_goals(thread_id TEXT,status TEXT)')
        db.execute('INSERT INTO thread_goals VALUES (?,?)',('thread',status))
    assert allowed('thread',codex_dir=tmp_path) is expected
    (tmp_path/'STOP').touch()
    assert not allowed('thread',tmp_path,tmp_path)


def test_missing_state_fails_closed_without_creating_a_database(tmp_path):
    assert not allowed('thread',codex_dir=tmp_path)
    assert not (tmp_path/'goals_1.sqlite').exists()
