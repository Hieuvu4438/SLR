import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from cico_replication_summary import summarize


def rows(values):
    return [dict(seed=s,endpoint_mean_delta=v,endpoint_delta={'T2V':v,'V2T':v},
                 selected_mean_delta=0.,accuracy_gate=False) for s,v in zip([42,1337,2026],values)]


def test_sample_variation_and_accuracy_separation():
    result=summarize(rows([1,2,3]))
    assert result['endpoint_delta_mean']==2 and result['endpoint_delta_sample_std']==1
    assert result['replication_gate'] and result['accuracy_gate_seed_count']==0


def test_no_missing_seed_or_duplicate_seed_summary():
    r=rows([1,2,3])
    with pytest.raises(ValueError):summarize(r[:2])
    r[-1]['seed']=42
    with pytest.raises(ValueError):summarize(r)


def test_negative_direction_prevents_replication_gate():
    r=rows([2,3,4])
    for item in r:item['endpoint_delta']['V2T']=-1
    assert not summarize(r)['replication_gate']
