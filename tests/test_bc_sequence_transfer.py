import copy

import pytest

import training.bc_sequence_transfer as study


def row(score, clear=0, piece=1):
    return dict(score=score, clear=clear, height=3, holes=0, new_holes=0,
                current='I', preview=['S', 'Z', 'O'], held=None, placed='I', hold=False,
                piece=piece, wells=[0]*10)


def test_common_window_uses_score_increments_and_real_death(monkeypatch):
    monkeypatch.setattr(study, 'FOLLOW', 2)
    base = dict(key=('branch', 7, 9, 0), rows=[row(110), row(120), row(130), row(140)],
                initial_score=100, game_over=False, snapshot_sha256='x')
    candidate = dict(key=('branch', 7, 9, 1), rows=[row(115), row(132)],
                     initial_score=100, game_over=True, snapshot_sha256='x')
    result = study.pair(base, candidate, 1)
    assert result['common_pieces'] == 2
    assert result['baseline']['score'] == 20
    assert result['intervention']['score'] == 32
    assert result['baseline']['death'] == 0
    assert result['intervention']['death'] == 1
    assert result['baseline']['pieces'] == 3
    # A baseline death beyond the intervention's observation limit is censored.
    base['game_over'] = True
    assert study.pair(base, candidate, 1)['baseline']['death'] == 0


def test_gate_counts_independent_seeds_and_rejects_death(monkeypatch):
    monkeypatch.setattr(study, 'SEEDS', tuple(range(6)))
    monkeypatch.setattr(study, 'STARTS', (1, 2, 3, 4))
    pairs = []
    for seed in study.SEEDS:
        for start in study.STARTS:
            for length in study.LENGTHS:
                a = dict(score=100, pieces=200, death=0, high=0, new_holes=0, four=0)
                b = dict(a, score=110, four=1)
                pairs.append(dict(seed=seed, start=start, length=length, common_pieces=200,
                                  baseline=a, intervention=b))
    result = study.decide(pairs)
    assert result['passed'] and result['selected_length'] == 16
    damaged = copy.deepcopy(pairs)
    for p in damaged:
        p['intervention']['death'] = 1
    assert not study.decide(damaged)['passed']
    # Repeated states from a single seed cannot supply cross-seed evidence.
    one_seed = [p for p in pairs if p['seed'] == 0]
    assert not study.decide(one_seed)['passed']


def test_future_labels_distinguish_boundary_death_and_censoring(monkeypatch):
    monkeypatch.setattr(study, 'FOLLOW', 2)
    monkeypatch.setattr(study, 'LENGTHS', (1,))
    rows = [row(100, 4, 1), row(110, piece=2), row(120, piece=3)]
    for r in rows:
        r['inference_seconds'] = .001
    source = dict(key=('ppo', 7, 0, 0), initial_score=0, rows=rows, game_over=True, capped=False)
    result = study.native_analysis(source)
    assert len(result['windows']['1']['death_within_2048']) == 1
    assert not result['windows']['1']['survived_2048']
    source.update(game_over=False, capped=True, rows=rows[:2])
    result = study.native_analysis(source)
    assert len(result['windows']['1']['right_censored']) == 1


def test_atomic_checksum_and_exact_interrupted_resume(tmp_path, monkeypatch):
    if not study.BC.exists():
        pytest.skip('Frozen models unavailable')
    monkeypatch.setattr(study, 'RUN', tmp_path)
    monkeypatch.setattr(study, 'CAP', 24)
    monkeypatch.setattr(study, 'STARTS', (4,))
    monkeypatch.setattr(study, 'CHECKPOINT_INTERVAL', 4)
    monkeypatch.setattr(study, 'IDENTITY', 'test')
    monkeypatch.setattr(study, 'guard', lambda: None)
    study.torch.set_num_threads(1)
    monkeypatch.setattr(study, 'MODELS', study.load_models())
    uninterrupted = study.read_blob(study.run_job('bc', 93))
    study.job_path('bc', 93).unlink()
    original_move = study.move
    def interrupt(env, source):
        if env.pieces == 6:
            raise InterruptedError('deliberate checkpoint test')
        return original_move(env, source)
    monkeypatch.setattr(study, 'move', interrupt)
    with pytest.raises(InterruptedError):
        study.run_job('bc', 93)
    monkeypatch.setattr(study, 'move', original_move)
    resumed = study.read_blob(study.run_job('bc', 93))
    strip = lambda rows: [{k:v for k,v in r.items() if k != 'inference_seconds'} for r in rows]
    assert strip(uninterrupted['rows']) == strip(resumed['rows'])
    assert uninterrupted['final_state']['core'].get_public_observation() == resumed['final_state']['core'].get_public_observation()
    path = tmp_path / 'corrupt.json'
    study.record(path, {'score': 1})
    path.write_text(path.read_text().replace('"score": 1', '"score": 2'))
    with pytest.raises(RuntimeError, match='checksum'):
        study.read_record(path)
