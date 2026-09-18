import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from extraction_resume import acquire_lock, atomic_json, digest, preserve_orphans, verify_item
from inventory import sha


def test_lease_rejects_second_owner(tmp_path):
    owner = acquire_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match='already owns'):
            acquire_lock(tmp_path)
    finally:
        owner.close()
    acquire_lock(tmp_path).close()


def test_resume_checks_inputs_outputs_and_contract(tmp_path):
    pose, rgb, raw = [tmp_path / n for n in ['pose.pkl', 'rgb.pkl', 'raw.mp4']]
    for path in [pose, rgb, raw]:
        path.write_bytes(path.name.encode())
    meta = tmp_path / 'metadata.json'
    assert not verify_item(meta, pose, rgb, raw, 'recipe')
    values = dict(input_contract_sha256='recipe', raw_sha256=sha(raw),
                  pose_sha256=sha(pose), rgb_sha256=sha(rgb), native_loader_smoke_pass=True)
    atomic_json(meta, values)
    assert verify_item(meta, pose, rgb, raw, 'recipe')
    with pytest.raises(ValueError):
        verify_item(meta, pose, rgb, raw, 'different_recipe')
    raw.write_bytes(b'changed')
    with pytest.raises(ValueError):
        verify_item(meta, pose, rgb, raw, 'recipe')
    values['raw_sha256'] = sha(raw)
    atomic_json(meta, values)
    rgb.write_bytes(b'corrupted')
    with pytest.raises(ValueError):
        verify_item(meta, pose, rgb, raw, 'recipe')


def test_orphans_are_preserved_not_deleted(tmp_path):
    folder = tmp_path / 'pose'
    folder.mkdir()
    pose = folder / 'sample.pkl'
    pose.write_bytes(b'partial artifact')
    preserve_orphans(tmp_path, 'sample', [pose])
    saved = list((tmp_path / 'orphaned').rglob('sample.pkl'))
    assert len(saved) == 1 and saved[0].read_bytes() == b'partial artifact'
    assert not pose.exists()


def test_contract_digest_and_atomic_json(tmp_path):
    assert digest({'a': 1, 'b': 2}) == digest({'b': 2, 'a': 1})
    path = tmp_path / 'run.json'
    atomic_json(path, {'status': 'running'})
    atomic_json(path, {'status': 'completed'})
    assert json.loads(path.read_text()) == {'status': 'completed'}
    assert not list(tmp_path.glob('*.partial-*'))
