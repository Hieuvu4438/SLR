"""AS-C36 exact input equality at FP32-loader and native-FP16 boundaries."""
from collections import Counter, defaultdict
import hashlib
import json
import os
import time
import traceback

import torch

from slr_common.data.cico_dataset import CiCoFeatureDataset
from .common import ROOT, dump, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def input_digest(h, valid):
    assert h.ndim == 2 and valid.shape == (len(h),) and valid.dtype == torch.bool
    assert torch.isfinite(h).all()
    x = h.detach().cpu().contiguous().clone()
    x[x == 0] = 0  # canonicalize signed zero, consistent with torch.equal
    digest = hashlib.sha256()
    digest.update(json.dumps([str(x.dtype), list(x.shape), list(valid.shape)]).encode())
    digest.update(x.numpy().tobytes())
    digest.update(valid.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def scoped_groups(entries, field):
    groups = defaultdict(list)
    for e in entries:
        groups[e[field]].append(e)
    def summaries(subsets):
        return {'group_n': len(subsets), 'row_n': sum(map(len, subsets)),
                'pair_n': sum(len(g)*(len(g)-1)//2 for g in subsets),
                'groups': [[{'split': e['split'], 'index': e['index'], 'pair_id': e['pair_id']} for e in g] for g in subsets]}
    result = {}
    for split in ('train', 'dev'):
        subsets = [[e for e in group if e['split'] == split] for group in groups.values()]
        result[split] = summaries([g for g in subsets if len(g) > 1])
    cross = [g for g in groups.values() if len({e['split'] for e in g}) > 1]
    result['cross_split'] = summaries(cross)
    result['cross_split']['pair_n'] = sum(sum(e['split'] == 'train' for e in g)*sum(e['split'] == 'dev' for e in g) for g in cross)
    return result, [g for g in groups.values() if len(g) > 1]


def main():
    path = OUT/'AS-C36-VISUAL-INPUT_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(4)
    started = time.time()
    result = {'experiment_id': 'AS-C36-VISUAL-INPUT', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C36_protocol.md'),
              'method_go': False, 'encoder_updates': 0, 'test_loaded': False,
              'splits': {}, 'entries': []}
    dump(path, result)
    try:
        cp = ROOT/'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
        result['checkpoint_sha256'] = sha(cp)
        assert result['checkpoint_sha256'] == 'ee45bd2dcad57ff4237eaeb2fc326d288698400c13aac0e51d9df01496b36351'
        state = torch.load(cp, map_location='cpu', weights_only=True, mmap=True)
        dtype = state['model']['core.clip.visual.conv1.weight'].dtype
        assert dtype == torch.float16
        result['native_visual_dtype'] = str(dtype)
        del state
        result['source_hashes'] = {str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT/'shared/slr_common/data/cico_dataset.py', ROOT/'shared/slr_common/data/views.py',
            ROOT/'third_party/SLRT/CiCo/CLCL/modules/module_clip.py')}
        datasets = {}
        for split in ('train', 'dev'):
            manifest = ROOT/f'artifacts/manifests/ph_{split}.jsonl'
            data = CiCoFeatureDataset(manifest, feature_len=64, alpha=.9, split=split)
            datasets[split] = data
            hist, zero, invalid_nonzero = Counter(), 0, 0
            for i in range(len(data)):
                if time.time()-started > 300:
                    raise TimeoutError('AS-C36 timeout300s')
                item = data[i]
                h, valid = item['h'], item['valid']
                assert h.shape == (64, 1024) and h.dtype == torch.float32
                assert torch.isfinite(h).all() and valid.any()
                native = h.to(dtype)
                assert torch.isfinite(native).all()
                length = int(valid.sum())
                hist[length] += 1
                zero += int(h[valid].eq(0).all())
                invalid_nonzero += int(h[~valid].ne(0).any())
                record = data.records[i]
                result['entries'].append({'split': split, 'index': i, 'pair_id': item['pair_id'],
                    'video_id': item['video_id'], 'valid_n': length, 'dense_length': record.dense_length,
                    'view_hash': item['view_hash'], 'fp32_input_sha256': input_digest(h, valid),
                    'fp16_input_sha256': input_digest(native, valid),
                    'aware_path': str(record.feature_aware), 'agnostic_path': str(record.feature_agnostic),
                    'aware_file_sha256': sha(record.feature_aware), 'agnostic_file_sha256': sha(record.feature_agnostic)})
                if (i+1) % 1000 == 0:
                    print(json.dumps({'split': split, 'done': i+1, 'wall_seconds': time.time()-started}), flush=True)
            result['splits'][split] = {'n': len(data), 'manifest_sha256': sha(manifest),
                'valid_length_histogram': dict(sorted(hist.items())), 'all_zero_valid_input_n': zero,
                'nonzero_padding_row_n': invalid_nonzero}
            dump(path, result)
        result['collisions'] = {}
        for precision, dtype in [('fp32', torch.float32), ('fp16', torch.float16)]:
            summary, groups = scoped_groups(result['entries'], f'{precision}_input_sha256')
            count = 0
            for group in groups:
                reference = datasets[group[0]['split']][group[0]['index']]
                for e in group[1:]:
                    current = datasets[e['split']][e['index']]
                    assert torch.equal(reference['valid'], current['valid'])
                    assert torch.equal(reference['h'].to(dtype), current['h'].to(dtype))
                    count += 1
            result['collisions'][precision] = {'scopes': summary, 'reloaded_equal_comparisons': count,
                                               'distinct_inputs': len({e[f'{precision}_input_sha256'] for e in result['entries']})}
        result['shared_feature_paths'] = {k: scoped_groups(result['entries'], k)[0] for k in ('aware_path', 'agnostic_path')}
        result.update(status='completed', wall_seconds=time.time()-started,
                      source_file_rehash_n=2*len(result['entries']))
        dump(path, result)
        print(json.dumps({'status': result['status'], 'collisions': result['collisions'],
                          'wall_seconds': result['wall_seconds'], 'source_file_rehash_n': result['source_file_rehash_n']}, indent=2), flush=True)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
