"""AS-C35 shared orthogonal coordinate control, TRAIN-only map fitting."""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
from .checkpoint_cross_probe import encode, score
from .common import ART, ROOT, dump, metrics, ranks, rows, sha
from .encoder_grid_probe import comparison, grid_aggregates
from .seed_ensemble_probe import OUT, SEEDS


def anchors(video, text):
    valid = video.mask == 0
    pooled = (F.normalize(video.tokens.float(), dim=-1)*valid[..., None]).sum(1)/valid.sum(1)[:, None]
    return torch.stack((F.normalize(pooled, dim=-1), F.normalize(text.cls.float(), dim=-1)), 1)


def fit_orthogonal(source, target):
    x, y = source.reshape(-1, source.shape[-1]).double(), target.reshape(-1, target.shape[-1]).double()
    u, singular, vh = torch.linalg.svd(x.T@y, full_matrices=False)
    q = u@vh
    err = float((q.T@q-torch.eye(q.shape[0], dtype=q.dtype)).abs().max())
    assert err < 1e-10
    return q, {'orthogonality_max_error': err, 'smallest_singular_value': float(singular[-1]),
               'anchor_cosine_before': float((x*y).sum(-1).mean()),
               'anchor_cosine_after': float(((x@q)*y).sum(-1).mean())}


def rotate(enc, q):
    return type(enc)(enc.mask, enc.tokens.float()@q, enc.cls.float()@q)


@torch.inference_mode()
def main():
    path = OUT/'AS-C35-GAUGE_run.json'
    if path.exists():
        raise FileExistsError(path)
    started = time.time()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {'experiment_id': 'AS-C35-GAUGE', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C35_protocol.md'),
              'encoder_updates': 0, 'map_fit_split': 'train', 'test_loaded': False, 'method_go': False,
              'maps': {}, 'variants': {}}
    dump(path, result)
    try:
        parent_path = OUT/'AS-C34-ENCODER-GRID_run.json'
        parent = json.loads(parent_path.read_text())
        assert parent['status'] == 'completed'
        result['parent_run_sha256'] = sha(parent_path)
        result['manifest_sha256'] = {s: sha(ROOT/f'artifacts/manifests/ph_{s}.jsonl') for s in ('train', 'dev')}
        assert result['manifest_sha256']['dev'] == parent['manifest_sha256']
        train_anchors, dev_enc, bridges = [], [], []
        for seed in SEEDS:
            cp = ROOT/f'runs/ph_base_b512_s{seed}/checkpoints/best_dev.pt'
            assert sha(cp) == parent['encoders'][str(seed)]['checkpoint_sha256']
            cfgpath = cp.parents[1]/'resolved_config.yaml'
            assert sha(cfgpath) == parent['encoders'][str(seed)]['config_sha256']
            config = yaml.safe_load(cfgpath.read_text())
            cico_root = ROOT/config['upstream']['cico_root']
            sys.path.insert(0, str(cico_root))
            saved = torch.load(cp, weights_only=True, map_location='cpu', mmap=True)
            state = {k.removeprefix('core.'): v for k, v in saved['model'].items() if k.startswith('core.')}
            core = _load_cico_core_from_state(config, state, cico_root=cico_root, device='cuda')
            core.eval().requires_grad_(False)
            bridge = CiCoBridge(core)
            bridges.append(bridge)
            for split in ('train', 'dev'):
                if time.time()-started > 300:
                    raise TimeoutError('AS-C35 timeout300s')
                data = CiCoFeatureDataset(ROOT/f'artifacts/manifests/ph_{split}.jsonl', feature_len=64, alpha=.9, split=split)
                loader = DataLoader(data, batch_size=128, num_workers=4, shuffle=False,
                                    collate_fn=CiCoCollator(load_cico_tokenizer(config), 32, augment=False))
                if split == 'dev':
                    v, t, ids = encode(core, loader)
                    dev_enc.append((v, t))
                else:
                    parts, ids = [], []
                    for b in loader:
                        v = bridge.encode_video(b['h'].cuda(), b['valid'].cuda())
                        t = bridge.encode_text(*(x.cuda() for x in b['clean_text']))
                        parts.append(anchors(v, t).cpu())
                        ids.extend(b['pair_id'])
                    train_anchors.append(torch.cat(parts))
                assert ids == [r['pair_id'] for r in rows(split)]
            print(json.dumps({'seed': seed, 'train_anchor_shape': list(train_anchors[-1].shape), 'dev_encoded': 519}), flush=True)
            del saved, state
        map_sets = {'identity': [torch.eye(512)]*3, 'aligned': [torch.eye(512)], 'shifted_target': [torch.eye(512)]}
        for name in ('aligned', 'shifted_target'):
            for i, seed in enumerate(SEEDS[1:], 1):
                target = train_anchors[0] if name == 'aligned' else train_anchors[0].roll(1, 0)
                q, audit = fit_orthogonal(train_anchors[i], target)
                actual = train_anchors[i].double().reshape(-1, 512)@q
                audit['true_pair_train_cosine_after'] = float((actual*train_anchors[0].reshape(-1, 512).double()).sum(-1).mean())
                da = anchors(*dev_enc[i]).cpu().double().reshape(-1, 512)
                db = anchors(*dev_enc[0]).cpu().double().reshape(-1, 512)
                audit['dev_anchor_cosine_before'] = float((da*db).sum(-1).mean())
                audit['dev_anchor_cosine_after'] = float(((da@q)*db).sum(-1).mean())
                p = ART/f'AS-C35-{name}-s{seed}_map.npy'
                if p.exists():
                    raise FileExistsError(p)
                np.save(p, q.numpy())
                audit['map_sha256'] = sha(p)
                result['maps'][f'{name}_{seed}'] = audit
                map_sets[name].append(q.float())
        dump(path, result)
        labels = [r['video_id'].rsplit('-', 1)[0] for r in rows('dev')]
        base = np.load(ART/'baseline_dev_scores.npy')
        original_mismatch = np.load(ART/'AS-C34-mismatched6_scores.npy')
        original_matched = np.load(ART/'AS-C34-matched3_scores.npy')
        for name, maps in map_sets.items():
            rotated = [tuple(rotate(e, q.cuda()) for e in pair) if name != 'identity' else pair for pair, q in zip(dev_enc, maps)]
            grid = [[None]*3 for _ in range(3)]
            entry = {'cells': {}, 'aggregates': {}}
            for i, vi in enumerate(SEEDS):
                for j, ti in enumerate(SEEDS):
                    if time.time()-started > 300:
                        raise TimeoutError('AS-C35 timeout300s')
                    s = score(bridges[i], rotated[i][0], rotated[j][1])
                    original_path = ART/f'AS-C34-video_{vi}_text_{ti}_scores.npy'
                    assert sha(original_path) == parent['cells'][f'video_{vi}_text_{ti}']['score_sha256']
                    original = np.load(original_path)
                    parity = {'max_abs_delta': float(np.abs(s-original).max()),
                              'rank_changes': {d: int((ranks(s)[d] != ranks(original)[d]).sum()) for d in ('T2V', 'V2T')}}
                    if name == 'identity':
                        assert np.array_equal(s, original)
                    if i == j:
                        assert parity['max_abs_delta'] <= 5e-5 and not any(parity['rank_changes'].values()), parity
                    grid[i][j] = s
                    p = ART/f'AS-C35-{name}-v{vi}-t{ti}_scores.npy'
                    if p.exists():
                        raise FileExistsError(p)
                    np.save(p, s)
                    entry['cells'][f'v{vi}_t{ti}'] = {'metrics': metrics(s, base), 'parity': parity, 'score_sha256': sha(p)}
            for key, s in grid_aggregates(grid).items():
                if key not in ('matched3', 'mismatched6', 'all9'):
                    continue
                entry['aggregates'][key] = {'metrics': metrics(s, base),
                    'versus_original_mismatched6': comparison(s, original_mismatch, labels, f'{name}/{key} minus original mismatched6'),
                    'versus_original_matched3': comparison(s, original_matched, labels, f'{name}/{key} minus original matched3')}
                p = ART/f'AS-C35-{name}-{key}_scores.npy'
                if p.exists():
                    raise FileExistsError(p)
                np.save(p, s)
                entry['aggregates'][key]['score_sha256'] = sha(p)
            result['variants'][name] = entry
            dump(path, result)
            print(json.dumps({'variant': name, 'mean_R1': {k: v['metrics']['official_mean_R1'] for k, v in entry['aggregates'].items()}}), flush=True)
        a = result['variants']['aligned']['aggregates']['mismatched6']
        n = result['variants']['shifted_target']['aggregates']['mismatched6']
        result['coordinate_explanation_lead'] = (a['versus_original_mismatched6']['mean_pp'] >= .5
            and a['versus_original_mismatched6']['lower'] > 0
            and a['metrics']['official_mean_R1']-n['metrics']['official_mean_R1'] >= .5)
        result.update(status='completed', wall_seconds=time.time()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(path, result)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
