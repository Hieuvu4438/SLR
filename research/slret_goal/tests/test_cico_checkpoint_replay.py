import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from cico_checkpoint_replay import require_exact


def test_replay_gate_rejects_score_rank_and_id_changes():
    matrix=np.eye(3,dtype=np.float32)
    metrics=dict(T2V={'cols':[0,0,0]},V2T={'cols':[0,0,0]},gallery={'videos':3,'texts':3},
                 metric_kernel='native',id_hashes={'videos':'v','texts':'t'})
    require_exact(matrix,metrics,matrix.copy(),copy.deepcopy(metrics))
    changed=matrix.copy()
    changed[0,1]=1e-8
    with pytest.raises(ValueError):require_exact(changed,metrics,matrix,metrics)
    for key,value in [('T2V',{'cols':[1,0,0]}),('id_hashes',{'videos':'bad','texts':'t'})]:
        changed=copy.deepcopy(metrics)
        changed[key]=value
        with pytest.raises(ValueError):require_exact(matrix,changed,matrix,metrics)
