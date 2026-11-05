import pytest

from training.evaluation.audit_reference_retention import aggregate, distribution_metrics


def test_masked_kl_and_argmax_detect_drift_even_with_small_reference_margin():
    same = distribution_metrics([0,0.51,0.49], [0,0.51,0.49], [1,2])
    flip = distribution_metrics([0,0.51,0.49], [0,0.49,0.51], [1,2])
    assert same['top1_same'] and same['masked_kl_candidate_to_reference'] == 0
    assert not flip['top1_same'] and flip['top5_retained']
    assert flip['reference_margin'] == pytest.approx(0.02)
    assert flip['masked_kl_candidate_to_reference'] > 0
    assert aggregate([same,flip])['top1_agreement'] == 0.5


@pytest.mark.parametrize('candidate', [[0.01,0.5,0.49], [0,float('nan'),1], [0,-0.1,1.1], [0,0.3,0.3]])
def test_invalid_or_unmasked_probabilities_are_rejected(candidate):
    with pytest.raises(ValueError):
        distribution_metrics([0,0.5,0.5],candidate,[1,2])


def test_single_legal_action_and_empty_group():
    assert distribution_metrics([0,1],[0,1],[1])['reference_margin'] == 1
    assert aggregate([]) == {'states':0}


def test_ties_follow_global_action_ids_even_if_sparse_rows_are_unsorted():
    result = distribution_metrics([0,0.5,0.5],[0,0.5,0.5],[2,1])
    assert result['reference_action'] == result['candidate_action'] == 1
    assert result['top1_same']
