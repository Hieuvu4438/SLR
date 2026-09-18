"""Read-only asset census; never load test labels or feature contents.

Generated reports and hashes are local research artifacts, not dataset copies.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/slret_goal/inventory'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    t = time.time()
    report = {'start_unix': t, 'pid': os.getpid(), 'test_contents_read': False,
              'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT)
    (OUT / 'preexisting_worktree.diff').write_bytes(diff)
    report['diff_sha256'] = hashlib.sha256(diff).hexdigest()
    report['resources'] = {key: subprocess.check_output(cmd, text=True) for key, cmd in {
        'gpu': ['nvidia-smi'], 'memory': ['free', '-h'], 'disk': ['df', '-h', str(ROOT)]}.items()}
    report['environment'] = {p.metadata['Name']: p.version for p in importlib.metadata.distributions()}
    report['source_hashes'] = {str(p.relative_to(ROOT)): sha(p) for base in
        ['third_party/SEDS', 'third_party/SLRT/CiCo/CLCL', 'third_party/UPRet']
        for p in (ROOT / base).rglob('*.py') if not any(x in p.parts for x in ['new_experiment', 'result_eval'])}
    report['checkpoints'] = {}
    for p in list((ROOT / 'third_party/SEDS/ckpt').glob('*')) + list((ROOT / 'third_party/SLRT/CiCo/CLCL/chpt').glob('*')):
        report['checkpoints'][str(p.relative_to(ROOT))] = {'resolved': str(p.resolve()),
            'exists': p.is_file(), **({'bytes': p.stat().st_size, 'sha256': sha(p)} if p.is_file() else {})}
    report['datasets'] = {}
    for dataset, data_dir, asset_dir in [('ph', 'data_ph', 'PHOENIX-2014-T'), ('csl', 'data_csl', 'CSL'), ('h2s', 'data_h2', 'How2Sign')]:
        base = ROOT / 'third_party/SEDS'
        train_path, dev_path = base / data_dir / 'train.pkl', base / data_dir / 'dev.pkl'
        train = pickle.load(train_path.open('rb'))
        dev = pickle.load(dev_path.open('rb')) if dev_path.exists() else {}
        # The manifest carries official dev IDs, not feature provenance for SEDS.
        manifest = ROOT / f'artifacts/manifests/{dataset}_dev.jsonl'
        rows = [json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else []
        info = {'train_rows': len(train), 'dev_pickle_rows': len(dev),
            'train_dev_pickle_key_overlap': len(set(train) & set(dev)),
            'train_sha256': sha(train_path), 'dev_sha256': sha(dev_path) if dev_path.exists() else None,
            'canonical_dev_manifest': str(manifest) if rows else None,
            'canonical_dev_manifest_sha256': sha(manifest) if rows else None,
            'canonical_dev_rows': len(rows), 'canonical_dev_train_key_overlap': len(set(train) & {r['video_id'] for r in rows})}
        assets = base / 'datasets' / asset_dir
        info['top_level_assets'] = [str(p.relative_to(base)) for p in assets.iterdir()]
        if dataset == 'ph':
            pose = assets / 'features/RTM_Keypoints'
            rgb = assets / 'features/I3D_features'
            info['dev_coverage'] = {'pose': sum((pose / (r['video_id'] + '.pkl')).is_file() for r in rows),
                'rgb_dev': sum((rgb / 'dev' / (r['video_id'] + '.pkl')).is_file() for r in rows),
                'rgb_train': sum((rgb / 'train' / (r['video_id'] + '.pkl')).is_file() for r in rows)}
        report['datasets'][dataset] = info
    report['wall_seconds'] = time.time() - t
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(OUT / 'report.json'), 'datasets': report['datasets'], 'wall_seconds': report['wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
