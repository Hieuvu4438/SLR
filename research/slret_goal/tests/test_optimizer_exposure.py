import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from optimizer_exposure import count_moments


def test_counts_both_optimizer_formats_and_denominators():
    result=count_moments({0:dict(next_m=torch.tensor([0,1,2],dtype=torch.float16),
                                 next_v=torch.tensor([0,0,1],dtype=torch.float16)),
                          1:dict(exp_avg=torch.tensor([0.]),exp_avg_sq=torch.tensor([0.]))})
    assert result['torch.float16']['fraction_all']==1/3
    assert result['torch.float16']['fraction_nonzero_first']==1/2
    assert result['torch.float32']['fraction_nonzero_first'] is None


def test_reject_mismatched_shapes():
    with pytest.raises(ValueError):
        count_moments({0:dict(next_m=torch.zeros(2),next_v=torch.zeros(3))})
