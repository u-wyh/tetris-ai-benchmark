"""A rejected candidate cannot win a parameter screen or its preservation tie."""

from pathlib import Path
import sys
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_learning_rate_control import select_candidate, run


def row(case,beta,ratio,eligible=True):
    return dict(case=case,parameter_value=beta,eligible_for_longer_development=eligible,
                comparison=dict(mean_observed_score_ratio=ratio))


def test_stronger_preservation_wins_only_within_the_predeclared_score_tie():
    cases=[row('weak',0.03,1.0),row('strong',0.3,0.997)]
    assert select_candidate(cases,prefer_higher=True)=='strong'
    cases[1]['comparison']['mean_observed_score_ratio']=0.994
    assert select_candidate(cases,prefer_higher=True)=='weak'


def test_rejected_candidate_never_wins_despite_high_score():
    assert select_candidate([row('failed',0.3,2.0,False)],True) is None
    assert select_candidate([row('valid',0.03,0.999),row('failed',0.3,2.0,False)],True)=='valid'


def test_restarting_completed_parameter_screen_does_not_start_work(tmp_path):
    (tmp_path/'summary.json').write_text('{}')
    (tmp_path/'status.json').write_text(json.dumps(dict(stage='completed_screen')))
    run(tmp_path)
    assert not (tmp_path/'queue.lock').exists()
