import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from cico_numeric_audit import select,check_rows


def test_guarded_selector_keeps_init_on_tie_and_direction_loss():
    ev={'0':{'T2V':{'R1':70},'V2T':{'R1':72}},
        '13':{'T2V':{'R1':73},'V2T':{'R1':71}},
        '26':{'T2V':{'R1':70.5},'V2T':{'R1':71.5}}}
    assert select(ev)['step']==0
    ev['39']={'T2V':{'R1':71},'V2T':{'R1':72}}
    assert select(ev)['step']==39


def test_paired_exposure_and_initial_gradient():
    a=[dict(step=i+1,epoch=i//13,ids=list(range(512)),input_sha256=str(i),lr=[.1],rng_digest='r',
            loss=.2,gradient_norm=.3,gradient_sha256='g') for i in range(260)]
    check_rows(a,copy.deepcopy(a))
    b=copy.deepcopy(a)
    b[0]['gradient_sha256']='bad'
    with pytest.raises(ValueError):check_rows(a,b)
    b=copy.deepcopy(a)
    b[17]['ids'].reverse()
    with pytest.raises(ValueError):check_rows(a,b)


def test_csl_240_step_exposure():
    a=[dict(step=i+1,epoch=i//12,ids=list(range(512)),input_sha256=str(i),lr=[.1],rng_digest='r',
            loss=.2,gradient_norm=.3,gradient_sha256='g') for i in range(240)]
    check_rows(a,copy.deepcopy(a),steps_per_epoch=12)
    with pytest.raises(ValueError):check_rows(a,a)
