import numpy as np
import pytest
from slr_common.evaluation.cico_eval import evaluate_score_matrix
from .clean_train_calibration import adequacy, compact_metrics, lr_factor


def test_compact_official_ties_and_strict_errors():
    s = np.array([[2, 2, 0], [2, 2, 3], [0, 1, 2]], dtype=np.float32)
    ids = ['a', 'b', 'c']
    maps = {x: [x] for x in ids}
    expected = evaluate_score_matrix(s, video_ids=ids, text_ids=ids, video_to_text=maps, text_to_video=maps)
    actual = compact_metrics(s, [(1,), (1,), (2,)])
    for d in ('T2V', 'V2T'):
        for k in (1, 5, 10):
            assert actual[d][f'R{k}'] == expected[d][f'R{k}']
        assert actual[d]['strict_different_text_error_n'] == 1


def test_schedule_and_gate_boundaries():
    assert lr_factor(1) == .01
    assert lr_factor(100) == 1
    assert lr_factor(1000) == 0
    assert 0 < lr_factor(500) < 1
    initial = {'held': {'mean_R1': 45}}
    final = {'fit': {d: {'R1': 80} for d in ('T2V', 'V2T')},
             'held': {d: {'R1': 50, 'strict_different_text_error_n': 50} for d in ('T2V', 'V2T')}}
    final['held']['mean_R1'] = 50
    assert adequacy(initial, final)['diagnostic_adequacy']
    assert not adequacy(initial, final)['method_go']
    final['held']['V2T']['R1'] = 49.999
    assert not adequacy(initial, final)['learning_gate']
