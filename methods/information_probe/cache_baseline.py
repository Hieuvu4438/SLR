"""Replay the pinned PH baseline, then cache exact frozen contextual features.

Run from repository root: PYTHONPATH=shared:. python -m
methods.information_probe.cache_baseline --splits dev train
Only train/dev manifests are accepted. No method-specific objective is imported.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator
from slr_common.upstream.cico_bridge import CiCoBridge, TextEncoding, VideoEncoding
from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
from .common import ART, ROOT, dump, metrics, rows, sha

OUT = ROOT / 'docs/proposal7/evidence/autonomous_search'


def subset(cache, prefix, start, end, device='cuda'):
    cls = VideoEncoding if prefix == 'video' else TextEncoding
    return cls(*(cache[f'{prefix}_{k}'][start:end].to(device)
                 for k in ('mask', 'tokens', 'cls')))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+', choices=['train', 'dev'], default=['dev', 'train'])
    parser.add_argument('--batch', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    np.random.seed(42)
    ART.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    checkpoint = ROOT / 'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
    config_path = ROOT / 'runs/ph_base_b512_s42/resolved_config.yaml'
    config = yaml.safe_load(config_path.read_text())
    selection = json.loads((checkpoint.parents[1] / 'selection.json').read_text())
    digest = sha(checkpoint)
    assert digest == selection['checkpoint_sha256']
    report = {
        'experiment_id': 'AS-C01-R0', 'status': 'running', 'pid': os.getpid(),
        'start_unix': start, 'command': sys.argv,
        'git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
        'config_sha256': sha(config_path), 'seed': 42, 'updates': 0,
        'optimizer': None, 'learning_rate_schedule': None, 'batch': args.batch,
        'precision': 'checkpoint-native encoder, float32 contextual tokens and scores',
        'gpu': torch.cuda.get_device_name(), 'splits': {},
        'code_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
                        (ROOT / 'methods/information_probe').glob('*.py')},
        'feature_provenance': config['data'].get('feature_provenance',
                              'See pinned resolved_config.yaml and row sidecars'),
        'dev_selector': 'historical best-dev checkpoint; no new selection',
        'protocol': 'full paired gallery; upstream Filip inner padding; legacy directional ties',
    }
    dump(OUT / 'AS-C01-R0_run.json', report)
    try:
        raw = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        state = {k.removeprefix('core.'): v for k, v in raw['model'].items()
                 if k.startswith('core.')}
        cico_root = ROOT / config['upstream']['cico_root']
        sys.path.insert(0, str(cico_root))
        core = _load_cico_core_from_state(config, state, cico_root=cico_root, device='cuda')
        core.eval().requires_grad_(False)
        del raw, state
        bridge = CiCoBridge(core)
        report['frozen_parameters'] = sum(p.numel() for p in core.parameters())
        report['trainable_parameters'] = sum(p.numel() for p in core.parameters() if p.requires_grad)
        tok = load_cico_tokenizer(config)
        for split in args.splits:
            manifest = ROOT / f'artifacts/manifests/ph_{split}.jsonl'
            output = ART / f'frozen_{split}.pt'
            if output.exists():
                raise FileExistsError(f'Refusing to overwrite existing cache: {output}')
            records = rows(split)
            data = CiCoFeatureDataset(manifest, feature_len=64, alpha=.9, split=split)
            loader = DataLoader(data, batch_size=args.batch, num_workers=4,
                                collate_fn=CiCoCollator(tok, 32, augment=False), shuffle=False)
            parts = {f'{p}_{f}': [] for p in ('video', 'text') for f in ('mask', 'tokens', 'cls')}
            ids = []
            for batch in loader:
                v = bridge.encode_video(batch['h'].cuda(), batch['valid'].cuda())
                t = bridge.encode_text(*(x.cuda() for x in batch['clean_text']))
                for prefix, enc in [('video', v), ('text', t)]:
                    for field in ('mask', 'tokens', 'cls'):
                        value = getattr(enc, field).cpu()
                        assert torch.isfinite(value).all()
                        parts[f'{prefix}_{field}'].append(value)
                ids.extend(batch['pair_id'])
                print(json.dumps({'event': 'encode', 'split': split, 'done': len(ids),
                                  'total': len(records), 'seconds': time.time()-start}), flush=True)
            assert ids == [r['pair_id'] for r in records]
            cache = {k: torch.cat(v) for k, v in parts.items()}
            cache.update(ids=ids, checkpoint_sha256=digest, manifest_sha256=sha(manifest),
                         logit_scale=float(core.clip.logit_scale.exp()))
            temporary = output.with_suffix('.pt.tmp')
            torch.save(cache, temporary)
            temporary.replace(output)
            entry = {'n': len(ids), 'manifest_sha256': sha(manifest),
                     'cache_sha256': sha(output), 'cache_bytes': output.stat().st_size,
                     'shapes': {k: list(v.shape) for k, v in cache.items() if torch.is_tensor(v)}}
            if split == 'dev':
                n = len(ids)
                channels = np.empty((2, n, n), dtype=np.float32)
                for i in range(0, n, 128):
                    for j in range(0, n, 128):
                        a, b = bridge.score(subset(cache, 'video', i, i+128),
                                            subset(cache, 'text', j, j+128))
                        channels[0, i:i+128, j:j+128] = a.cpu().numpy()
                        channels[1, i:i+128, j:j+128] = b.cpu().numpy()
                scores = channels.mean(0)
                historical = np.load(ROOT / 'runs/ph_base_b512_s42/evaluation/dev/'
                                     'scores_video_x_text.npy')
                delta = float(np.max(np.abs(scores-historical)))
                measured = metrics(scores, historical)
                dump(OUT / 'AS-C01-R0_metrics.json', measured)
                np.save(ART / 'baseline_dev_channels.npy', channels)
                np.save(ART / 'baseline_dev_scores.npy', scores)
                entry['max_abs_historical_score_delta'] = delta
                entry['rank_parity'] = all(not any(measured[d]['rank_delta']) for d in ('T2V', 'V2T'))
                entry['official_mean_R1'] = measured['official_mean_R1']
                report['splits'][split] = entry
                dump(OUT / 'AS-C01-R0_run.json', report)
                print(json.dumps({'event': 'replay', **entry}), flush=True)
                if delta > 1e-3 or not entry['rank_parity']:
                    raise RuntimeError('R0 parity gate failed; do not proceed with training')
            report['splits'][split] = entry
            dump(OUT / 'AS-C01-R0_run.json', report)
            del cache, parts
        report['status'] = 'completed'
    except Exception:
        report['status'] = 'failed'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['wall_seconds'] = time.time()-start
        report['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        dump(OUT / 'AS-C01-R0_run.json', report)


if __name__ == '__main__':
    main()
