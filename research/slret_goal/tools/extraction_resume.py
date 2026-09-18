"""Fail-closed, per-video recovery for our generated feature artifacts only."""
import fcntl
import hashlib
import json
import os
from pathlib import Path

from inventory import sha


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def acquire_lock(out):
    handle = (Path(out) / '.extract.lock').open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('Extractor already owns this output directory')
    return handle


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.partial-{os.getpid()}')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def verify_item(meta_path, pose_path, rgb_path, raw_path, contract_sha):
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    expected = {'input_contract_sha256': contract_sha, 'raw_sha256': sha(raw_path),
                'pose_sha256': sha(pose_path), 'rgb_sha256': sha(rgb_path),
                'native_loader_smoke_pass': True}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Completed artifact failed verification: {meta_path}')
    return True


def preserve_orphans(out, vid, paths):
    """Preserve unpublished partial item files before recomputing this item."""
    out = Path(out).resolve()
    for path in paths:
        path = Path(path)
        if path.exists():
            if not path.resolve().is_relative_to(out):
                raise ValueError('Refusing recovery outside generated output root')
            destination = out / 'orphaned' / f'{vid}-{os.getpid()}' / path.parent.name / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            path.rename(destination)
