import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from cico_ph_test_analysis import contributions,cluster_interval


def test_tie_expansion_preserved_not_optimistic_scalar():
    rows=[dict(query_id='a',rank=0,official_tie_ranks=[0,1]),dict(query_id='b',rank=2,official_tie_ranks=[2])]
    assert contributions(rows,['a','b'],'T2V').tolist()==[[1,2],[0,1]]
    assert contributions(rows,['a','b'],'V2T').tolist()==[[1,1],[0,1]]
    with pytest.raises(ValueError):contributions(rows,['b','a'],'T2V')


def test_paired_cluster_null_and_constant_effect():
    values=np.zeros((3,2,2,4,2),dtype=np.int64)
    values[...,1]=1
    null=cluster_interval(values,['a','a','b','b'],draws=100)
    assert null['ci95']==[0,0] and null['cluster_count']==2
    values[:,1,:,:,0]=1
    positive=cluster_interval(values,['a','a','b','b'],draws=100)
    assert positive['ci95']==[100,100]
