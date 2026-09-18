import copy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from cico_locked_ph_test import TEST_SHA,verify_lock


def fixture():
    return dict(test_manifest='test',test_manifest_sha256=TEST_SHA,
                sources={'source':'sourcehash'},evidence={},assets={'feature':'featurehash'},
                models=[dict(name=n,checkpoint=n,checkpoint_sha256=n+'hash') for n in
                        ['initial',*[f'{a}_s{s}' for a in ['native','fp32'] for s in [42,1337,2026]]]])


def test_lock_rejects_asset_source_manifest_and_model_set_drift():
    lock=fixture()
    hashes={'test':TEST_SHA,'source':'sourcehash','feature':'featurehash',
            **{m['checkpoint']:m['checkpoint_sha256'] for m in lock['models']}}
    with patch('cico_locked_ph_test.sha',side_effect=lambda p:hashes[str(p)]):
        verify_lock(lock)
        for key in ['source','feature','test','initial']:
            original=hashes[key]
            hashes[key]='changed'
            with pytest.raises(ValueError):verify_lock(lock)
            hashes[key]=original
        partial=copy.deepcopy(lock)
        partial['models'].pop()
        with pytest.raises(ValueError):verify_lock(partial)
