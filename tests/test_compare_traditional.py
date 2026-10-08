import json

import pytest

from training.evaluation.compare_traditional import _bootstrap, _sign_test, compare
from training.evaluation.traditional import run as run_v1
from training.evaluation.traditional_v2 import run as run_v2
from training.traditional.v2 import V2Config


def test_paired_protocol_and_uncertainty(tmp_path):
    v1, v2 = tmp_path/'v1', tmp_path/'v2'
    run_v1(v1,[11,12],2)
    run_v2(v2,[11,12],2,V2Config(mode='hold'))
    paired = compare(v1,v2)
    assert [row['seed'] for row in paired['paired']] == [11,12]
    assert paired['max_pieces'] == 2
    assert paired['summary']['both_cap_count'] == 2
    assert paired['summary']['sign_test_two_sided_p'] == 1
    assert _bootstrap([0,0]) == [0,0]
    assert _sign_test(4,0) == .125
    config = json.loads((v2/'config.json').read_text())
    config['max_pieces'] = 3
    (v2/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='identical seeds and cap'):
        compare(v1,v2)
