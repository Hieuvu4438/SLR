"""AS-C38 unchanged-precision encoder tail padding, independent-query control."""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator
from slr_common.evaluation.cico_eval import evaluate_score_matrix
from slr_common.upstream.cico_bridge import CiCoBridge, TextEncoding, VideoEncoding
from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
from .checkpoint_cross_probe import encode, score
from .common import ART, ROOT, dump, metrics, ranks, rows, sha
from .encoder_grid_probe import comparison
from .seed_ensemble_probe import OUT, SEEDS, uniform_mean


def pad_rows(x, size=128):
    assert 0 < len(x) <= size
    if len(x) == size:
        return x
    return torch.cat((x, x[:1].expand(size-len(x), *x.shape[1:])), dim=0)


@torch.inference_mode()
def padded_encode(core, loader):
    bridge = CiCoBridge(core)
    videos, texts, ids = [], [], []
    for b in loader:
        n = len(b['h'])
        v = bridge.encode_video(pad_rows(b['h']).cuda(), pad_rows(b['valid']).cuda())
        t = bridge.encode_text(*(pad_rows(x).cuda() for x in b['clean_text']))
        videos.append(VideoEncoding(*(getattr(v, k)[:n] for k in ('mask', 'tokens', 'cls'))))
        texts.append(TextEncoding(*(getattr(t, k)[:n] for k in ('mask', 'tokens', 'cls'))))
        ids.extend(b['pair_id'])
    return (VideoEncoding(*(torch.cat([getattr(v, k) for v in videos]) for k in ('mask', 'tokens', 'cls'))),
            TextEncoding(*(torch.cat([getattr(t, k) for t in texts]) for k in ('mask', 'tokens', 'cls'))), ids)


def sensitivity(candidate, baseline):
    a, b = ranks(candidate), ranks(baseline)
    return {'max_score_abs_delta': float(np.abs(candidate-baseline).max()),
            'changed_score_n': int((candidate != baseline).sum()),
            'T2V_expanded_entries': int((candidate == candidate.diagonal()[None, :]).sum()),
            'directions': {d: {'changed_rank_indices': np.flatnonzero(a[d] != b[d]).tolist(),
                               'R1_gained_indices': np.flatnonzero((a[d] == 0) & (b[d] > 0)).tolist(),
                               'R1_lost_indices': np.flatnonzero((a[d] > 0) & (b[d] == 0)).tolist()}
                           for d in a}}


@torch.inference_mode()
def main():
    path = OUT/'AS-C38-BATCH-SHAPE_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    started = time.time()
    result = {'experiment_id': 'AS-C38-BATCH-SHAPE', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C38_protocol.md'),
              'updates': 0, 'test_loaded': False, 'method_go': False, 'encoders': {}, 'scores': {},
              'environment': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                              'gpu': torch.cuda.get_device_name()}}
    dump(path, result)
    try:
        parent_path = OUT/'AS-C32-ENSEMBLE_run.json'
        parent = json.loads(parent_path.read_text())
        assert sha(parent_path) == 'bb16708f342f6b6cba64aa8c2949df4e4034375967b95e6ceaf5ff18767a3f11'
        assert parent['status'] == 'completed'
        result['parent_run_sha256'] = sha(parent_path)
        rec = rows('dev')
        ids = [r['pair_id'] for r in rec]
        labels = [r['video_id'].rsplit('-', 1)[0] for r in rec]
        positives = {i: [i] for i in ids}
        manifest = ROOT/'artifacts/manifests/ph_dev.jsonl'
        assert sha(manifest) == parent['manifest_sha256'] and len(ids) == 519
        result['manifest_sha256'] = sha(manifest)
        result['source_hashes'] = {str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT/'shared/slr_common/data/cico_dataset.py', ROOT/'shared/slr_common/data/tokenize.py',
            ROOT/'shared/slr_common/upstream/cico_bridge.py', ROOT/'methods/information_probe/checkpoint_cross_probe.py',
            ROOT/'third_party/SLRT/CiCo/CLCL/modules/module_clip.py')}
        matrices = {}
        for seed in SEEDS:
            if time.time()-started > 300:
                raise TimeoutError('AS-C38 timeout300s')
            root = ROOT/f'runs/ph_base_b512_s{seed}'
            cp = root/'checkpoints/best_dev.pt'
            assert sha(cp) == parent['inputs'][str(seed)]['checkpoint_sha256']
            original_path = root/'evaluation/dev/scores_video_x_text.npy'
            assert sha(original_path) == parent['inputs'][str(seed)]['score_sha256']
            original = np.load(original_path)
            config = yaml.safe_load((root/'resolved_config.yaml').read_text())
            cico_root = ROOT/config['upstream']['cico_root']
            sys.path.insert(0, str(cico_root))
            saved = torch.load(cp, weights_only=True, map_location='cpu', mmap=True)
            state = {k.removeprefix('core.'): v for k, v in saved['model'].items() if k.startswith('core.')}
            core = _load_cico_core_from_state(config, state, cico_root=cico_root, device='cuda')
            core.eval().requires_grad_(False)
            assert core.clip.visual.conv1.weight.dtype == torch.float16
            data = CiCoFeatureDataset(manifest, feature_len=64, alpha=.9, split='dev')
            loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=4,
                                collate_fn=CiCoCollator(load_cico_tokenizer(config), 32, augment=False))
            v, t, actual = encode(core, loader)
            assert actual == ids
            bridge = CiCoBridge(core)
            replay = score(bridge, v, t)
            assert np.array_equal(replay, original)
            pv, pt, actual = padded_encode(core, loader)
            assert actual == ids
            padded = score(bridge, pv, pt)
            rv, rt, actual = padded_encode(core, loader)
            assert actual == ids
            entry = {'checkpoint_sha256': sha(cp), 'config_sha256': sha(root/'resolved_config.yaml'),
                     'original_score_replay_exact': True, 'logit_scale': float(core.clip.logit_scale.exp()),
                     'towers': {}}
            for name, old, new, repeat in [('video', v, pv, rv), ('text', t, pt, rt)]:
                assert torch.equal(old.mask, new.mask)
                for key in ('mask', 'tokens', 'cls'):
                    assert torch.equal(getattr(old, key)[:512], getattr(new, key)[:512])
                    assert torch.equal(getattr(new, key), getattr(repeat, key))
                delta = (new.tokens-old.tokens).abs()
                entry['towers'][name] = {'full512_exact': True, 'padded_repeat_exact': True,
                    'max_tail_token_abs_delta': float(delta[512:].max()),
                    'changed_tail_token_scalar_n': int(delta[512:].ne(0).sum())}
            assert np.array_equal(score(bridge, rv, rt), padded)
            entry['padded_score_repeat_exact'] = True
            entry['sensitivity'] = sensitivity(padded, original)
            result['encoders'][str(seed)] = entry
            matrices[f'original_{seed}'] = replay
            matrices[f'padded_{seed}'] = padded
            dump(path, result)
            print(json.dumps({'seed': seed, **entry}), flush=True)
            del core, bridge, saved, state, v, t, pv, pt, rv, rt
        for policy in ('original', 'padded'):
            matrices[f'{policy}_ensemble'] = uniform_mean([matrices[f'{policy}_{s}'] for s in SEEDS])
        previous_ensemble = ART/'AS-C32-uniform3_scores.npy'
        assert sha(previous_ensemble) == parent['ensemble_score_sha256']
        assert np.array_equal(matrices['original_ensemble'], np.load(previous_ensemble))
        for name, s in matrices.items():
            m = metrics(s, matrices['original_42'])
            official = evaluate_score_matrix(s, video_ids=ids, text_ids=ids,
                                            video_to_text=positives, text_to_video=positives)
            for d, key in [('T2V', 'official_T2V'), ('V2T', 'V2T')]:
                assert all(m[key][f'R{k}'] == official[d][f'R{k}'] for k in (1, 5, 10))
            target = ART/f'AS-C38-{name}_scores.npy'
            if target.exists():
                raise FileExistsError(target)
            np.save(target, s)
            result['scores'][name] = {'metrics': m, 'score_sha256': sha(target), 'official_validation_exact': True}
        pe, oe, pb = matrices['padded_ensemble'], matrices['original_ensemble'], matrices['padded_42']
        result['ensemble_sensitivity'] = sensitivity(pe, oe)
        result['bootstrap_vs_padded42'] = comparison(pe, pb, labels, 'padded ensemble minus padded seed42')
        result['bootstrap_vs_original_ensemble'] = comparison(pe, oe, labels, 'padded ensemble minus original ensemble')
        m, b = metrics(pe, pb), metrics(pb)
        ci = result['bootstrap_vs_padded42']
        gate = {'mean_gain_at_least_half_pp': ci['mean_pp'] >= .5,
                'bootstrap_lower_positive': ci['lower'] > 0,
                'R1_nonregression': all(m[d]['R1']-b[d]['R1'] >= -.25 for d in ('official_T2V', 'V2T')),
                'R5_R10_nonregression': all(m[d][f'R{k}']-b[d][f'R{k}'] >= -.5 for d in ('official_T2V', 'V2T') for k in (5, 10)),
                'persistent_ranks_improve_both': all(m[d]['persistent_mean_rank_delta'] < 0 for d in ('T2V', 'V2T'))}
        gate['diagnostic_lead'] = all(gate.values())
        result.update(status='completed', wall_seconds=time.time()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(), gate=gate,
                      actual_full_score_calls=9, independent_ensemble_replicates=1)
        dump(path, result)
        print(json.dumps({'status': result['status'], 'mean_R1': {k: v['metrics']['official_mean_R1'] for k, v in result['scores'].items()},
                          'ensemble_sensitivity': result['ensemble_sensitivity'], 'gate': gate,
                          'wall_seconds': result['wall_seconds']}, indent=2), flush=True)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
