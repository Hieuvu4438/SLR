import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from csl_test_assets import validate_feature
from inventory import sha


def test_feature_gate_rejects_drift_batch_nonfinite_and_identity(tmp_path):
    path=tmp_path/'feature.pkl'
    sidecar=path.with_suffix('.pkl.meta.json')
    values=np.ones((3,1024),dtype=np.float32)
    def save(batch=128):
        path.write_bytes(pickle.dumps({'feature':values}))
        sidecar.write_text(json.dumps(dict(feature_sha256=sha(path),
            source_video_sha256='source',recipe_sha256='recipe',
            checkpoint_sha256='checkpoint',effective_batch_size=batch,feature_shape=[3,1024])))
    def check(**kwargs):
        return validate_feature(path,source_sha='source',recipe_sha='recipe',
                                checkpoint_sha=kwargs.get('checkpoint','checkpoint'))
    save()
    assert check()[0]==(3,1024)
    with pytest.raises(ValueError,match='provenance'):
        check(checkpoint='changed')
    save(batch=64)
    with pytest.raises(ValueError,match='batch'):
        check()
    save()
    path.write_bytes(path.read_bytes()+b'drift')
    with pytest.raises(ValueError,match='hash'):
        check()
    values[0,0]=np.nan
    save()
    with pytest.raises(ValueError,match='values'):
        check()
