import copy
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from cico_locked_csl_test import NAMES, SELECTED_STEPS, verify_lock
from cico_csl_test_analysis import cluster_counts,cluster_interval


def test_csl_lock_rejects_assets_models_selection_and_steps():
    models=[]
    for name in NAMES:
        step=0 if name=='initial' else (SELECTED_STEPS[name.removeprefix('selected_')] if name.startswith('selected_') else 240)
        models.append(dict(name=name,step=step,checkpoint=name,checkpoint_sha256=name+'hash'))
    lock=dict(gallery=dict(videos=1176,texts=798),test_manifest='test',test_manifest_sha256='testhash',
        sources={'source':'sourcehash'},evidence={},assets={'feature':'featurehash'},models=models,
        selected_mapping={n:f'selected_{n}' if s else 'initial' for n,s in SELECTED_STEPS.items()})
    hashes={'test':'testhash','source':'sourcehash','feature':'featurehash',**{m['checkpoint']:m['checkpoint_sha256'] for m in models}}
    with patch('cico_locked_csl_test.sha',side_effect=lambda p:hashes[str(p)]):
        verify_lock(lock)
        for key in ['test','source','feature','initial']:
            prior=hashes[key];hashes[key]='changed'
            with pytest.raises(ValueError):verify_lock(lock)
            hashes[key]=prior
        for change in ['missing','step','selection']:
            broken=copy.deepcopy(lock)
            if change=='missing':broken['models'].pop()
            elif change=='step':broken['models'][0]['step']=-1
            else:broken['selected_mapping']['native_s42']='initial'
            with pytest.raises(ValueError):verify_lock(broken)


def test_grouped_counts_preserve_video_denominator_not_tie_expansion():
    query=dict(T2V=[dict(query_id='a',rank=0,official_tie_ranks=[0,1]),dict(query_id='b',rank=1)],
        V2T=[dict(query_id='v1',rank=0),dict(query_id='v2',rank=2),dict(query_id='v3',rank=0)])
    counts=cluster_counts(query,['v1','v2','v3'],['a','b'],dict(v1='a',v2='a',v3='b'))
    assert counts.tolist()==[[[1,1],[0,1]],[[1,2],[1,1]]]
    with pytest.raises(ValueError,match='order'):
        cluster_counts(query,['v2','v1','v3'],['a','b'],dict(v1='a',v2='a',v3='b'))


def test_paired_bootstrap_null_constant_and_unequal_cluster_sizes():
    base=np.array([[[1,1],[0,1]],[[1,2],[1,1]]])
    counts=np.tile(base,(3,2,1,1,1))
    assert cluster_interval(counts,draws=500)['ci95']==[0.,0.]
    counts[:,:,0,:,1]=1
    counts[:,0,:,:,0]=0
    counts[:,1,:,:,0]=counts[:,1,:,:,1]
    assert cluster_interval(counts,draws=500)['ci95']==[100.,100.]
    # Unequal video cluster sizes must use ratio-of-sums, not mean group accuracy.
    assert base[1,:,0].sum()/base[1,:,1].sum()==pytest.approx(2/3)
