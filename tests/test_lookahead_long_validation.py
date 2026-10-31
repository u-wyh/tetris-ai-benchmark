from pathlib import Path
import copy
import pickle
import numpy as np
import pytest
from tests.test_lookahead_ablation import models
import training.lookahead_long_validation as study


def setup(monkeypatch, models, path):
    monkeypatch.setattr(study,'MODELS',{'A':models[0],'BC':models[1]})
    monkeypatch.setattr(study,'IDENTITY','unit_identity')
    monkeypatch.setattr(study,'MAIN_CAP',6)
    monkeypatch.setattr(study,'CHECKPOINT_INTERVAL',2)
    monkeypatch.setattr(study,'RUN',path)
    monkeypatch.setattr(study,'guard',lambda:None)


def test_continuation_matches_uninterrupted_with_exact_prefix(models,tmp_path,monkeypatch):
    setup(monkeypatch,models,tmp_path)
    direct=Path(study.game('E',53,12,'direct'))
    parent=Path(study.game('E',53,6,'100k'))
    original=study.digest(parent);snapshot=study.read_blob(parent.with_suffix('.end.pkl'))
    assert snapshot['env']['finished'] and snapshot['env']['pieces']==6
    extended=Path(study.game('E',53,12,'200k'))
    assert study.digest(parent)==original
    with np.load(direct.with_suffix('.npz')) as a,np.load(extended.with_suffix('.npz')) as b,np.load(parent.with_suffix('.npz')) as c:
        for key in ('scores','lines','codes','root_codes','costs'):
            np.testing.assert_array_equal(a[key],b[key])
        for key in c.files:np.testing.assert_array_equal(c[key],b[key][:len(c[key])])
    a=study.read_blob(direct.with_suffix('.end.pkl'))['env']['core']
    b=study.read_blob(extended.with_suffix('.end.pkl'))['env']['core']
    assert pickle.dumps(a)==pickle.dumps(b)
    assert study.completed(extended,'E',53,12)['parent_100k_result_sha256']==original
    before=study.digest(extended);study.game('E',53,12,'200k');assert study.digest(extended)==before
    parent.with_suffix('.end.pkl').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError,match='snapshot checksum'):study.completed(parent,'E',53,6)


def test_atomic_interruption_and_identity_guard(models,tmp_path,monkeypatch):
    setup(monkeypatch,models,tmp_path)
    study.game('B',53,6,'100k')
    save=study.save_blob;count=0
    def interrupted(path,value):
        nonlocal count
        save(path,value);count+=1
        if count==2:raise RuntimeError('one simulated interrupted continuation')
    monkeypatch.setattr(study,'save_blob',interrupted)
    with pytest.raises(RuntimeError,match='simulated'):study.game('B',53,12,'200k')
    monkeypatch.setattr(study,'save_blob',save)
    checkpoint=tmp_path/'evaluation/200k/B/seed_53.checkpoint.pkl';saved=study.read_blob(checkpoint)
    bad=copy.deepcopy(saved);bad['state']['identity']='different_source';save(checkpoint,bad)
    with pytest.raises(RuntimeError,match='identity'):study.game('B',53,12,'200k')
    save(checkpoint,saved);extended=Path(study.game('B',53,12,'200k'));direct=Path(study.game('B',53,12,'direct'))
    with np.load(direct.with_suffix('.npz')) as a,np.load(extended.with_suffix('.npz')) as b:
        for key in ('scores','lines','codes','root_codes','costs'):np.testing.assert_array_equal(a[key],b[key])


def test_dead_parent_is_carried_without_any_new_decision(models,tmp_path,monkeypatch):
    setup(monkeypatch,models,tmp_path)
    parent=Path(study.game('E',53,6,'100k'))
    # Synthetic terminal fixture only; no research episode or training label.
    row=study.read_record(parent);saved=study.read_blob(parent.with_suffix('.end.pkl'))
    row.update(game_over=True,capped=False);saved['state'].update(game_over=True,capped=False)
    saved['env']['core'].phase='over';study.save_blob(parent.with_suffix('.end.pkl'),saved)
    row['snapshot_sha256']=study.digest(parent.with_suffix('.end.pkl'));study.record(parent,row)
    def forbidden(*args,**kw):raise AssertionError('A dead parent must never restart or infer')
    monkeypatch.setattr(study.AblationController,'decide',forbidden)
    extended=Path(study.game('E',53,12,'200k'));result=study.completed(extended,'E',53,12)
    assert result['pieces']==6 and result['game_over'] and not result['capped']
    assert result['already_dead_at_100k'] and result['inference_seconds']==row['inference_seconds']


def test_admission_is_not_statistical_safety_and_requires_extension():
    from training.lookahead_long_validation_report import decision
    p=dict(common_score_ratio=.998,ratio_ci95=[.995,1.001],survival_difference_ci95=[0,0],mean_episode_p95_ratio_ci95=[.96,.98])
    summary={'C':dict(mean_pieces=100000,deaths=0,early=0,p95_ms=37),'E':dict(mean_pieces=100000,deaths=0,early=0,p95_ms=39)}
    main=dict(complete=True,summary=summary,comparisons={'C_vs_E':p})
    extension=copy.deepcopy(main)
    r=decision(main,extension)
    assert r['engineering_candidate'] and r['survival_advantage']=='优势不确定'
    assert r['recommend_independent_confirmation']
    extension['complete']=False
    assert not decision(main,extension)['engineering_candidate']
    extension['complete']=True;extension['summary']['C']['deaths']=2
    assert decision(main,extension)['extension_opposition']
    assert not decision(main,extension)['recommend_independent_confirmation']
