import pytest

from training.env.bitboard_search_env import validate_search_backend


def test_search_default_preserves_formal_generator():
    assert validate_search_backend({}) == 'formal'
    assert validate_search_backend(dict(candidate_search_backend='formal')) == 'formal'


@pytest.mark.parametrize('patch', [{}, dict(candidate_policy=False),
    dict(fast_candidate_env=False), dict(candidate_feature_backend='official'),
    dict(failure_curriculum={'mode':'normal'}), dict(candidate_search_backend='unknown')])
def test_search_backend_cannot_be_silently_ignored(patch):
    preset=dict(candidate_policy=True, fast_candidate_env=True,
                candidate_feature_backend='bitboard', candidate_search_backend='bitboard')
    preset.update(patch)
    if not patch:
        assert validate_search_backend(preset) == 'bitboard'
    else:
        with pytest.raises(ValueError):
            validate_search_backend(preset)
