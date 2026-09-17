"""AS-C37 valid temporal slot equality with source-index/stream attribution."""
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
import time
import traceback

import torch

from slr_common.data.cico_dataset import CiCoFeatureDataset, load_dense_feature
from .common import ROOT, dump, sha
from .visual_input_inventory import input_digest

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def slot_groups(h, valid):
    assert h.ndim == 2 and valid.shape == (len(h),) and valid.dtype == torch.bool
    assert torch.isfinite(h).all()
    groups = defaultdict(list)
    x = h.detach().cpu().contiguous().clone()
    x[x == 0] = 0
    for slot in valid.nonzero().flatten().tolist():
        groups[x[slot].numpy().tobytes()].append(slot)
    return [g for g in groups.values() if len(g) > 1]


def describe(h, valid, dense_index, aware=None, agnostic=None):
    groups = slot_groups(h, valid)
    causes, marginal = Counter(), Counter()
    pairs = adjacent = 0
    for group in groups:
        assert aware is not None and agnostic is not None
        for a, b in combinations(group, 2):
            assert torch.equal(h[a], h[b])
            ia, ib = int(dense_index[a]), int(dense_index[b])
            assert ia >= 0 and ib >= 0
            ae = torch.equal(aware[ia], aware[ib])
            ge = torch.equal(agnostic[ia], agnostic[ib])
            # The dataset uses (1-alpha), not the decimal literal .1.
            fa = (1-.9)*aware[ia]+.9*agnostic[ia]
            fb = (1-.9)*aware[ib]+.9*agnostic[ib]
            fe = torch.equal(fa, fb)
            assert torch.equal(fa.to(h.dtype), h[a]) and torch.equal(fb.to(h.dtype), h[b])
            cause = ('same_dense_index' if ia == ib else
                     'both_streams_equal' if fe and ae and ge else
                     'fused_fp32_equal_not_both_streams' if fe else 'fp16_only_merger')
            if cause == 'fp16_only_merger':
                assert h.dtype == torch.float16
            causes[cause] += 1
            marginal['aware_equal_pair_n'] += int(ae)
            marginal['agnostic_equal_pair_n'] += int(ge)
            pairs += 1
            adjacent += int(b == a+1)
    n = int(valid.sum())
    return {'valid_n': n, 'unique_n': n-sum(len(g)-1 for g in groups),
            'excess_slot_n': sum(len(g)-1 for g in groups),
            'eligible_pair_n': n*(n-1)//2, 'repeated_pair_n': pairs,
            'adjacent_repeated_pair_n': adjacent, 'groups': groups,
            'pair_causes': dict(causes), 'source_marginal_counts': dict(marginal)}


def aggregate(entries, split, precision):
    items = [e[precision] for e in entries if e['split'] == split]
    keys = ('valid_n', 'unique_n', 'excess_slot_n', 'eligible_pair_n',
            'repeated_pair_n', 'adjacent_repeated_pair_n')
    result = {k: sum(x[k] for x in items) for k in keys}
    result.update(row_n=len(items), repeated_row_n=sum(x['repeated_pair_n'] > 0 for x in items),
                  at_least_two_valid_row_n=sum(x['valid_n'] >= 2 for x in items))
    for key in ('pair_causes', 'source_marginal_counts'):
        c = Counter()
        for x in items:
            c.update(x[key])
        result[key] = dict(c)
    return result


def main():
    path = OUT/'AS-C37-TEMPORAL-SLOTS_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(4)
    started = time.time()
    previous = OUT/'AS-C36-VISUAL-INPUT_run.json'
    pin = json.loads((OUT/'AS-C37_inputs.json').read_text())
    assert sha(previous) == pin['AS-C36_run_sha256']
    old = json.loads(previous.read_text())
    assert old['status'] == 'completed'
    result = {'experiment_id': 'AS-C37-TEMPORAL-SLOTS', 'status': 'running',
              'pid': os.getpid(), 'code_sha256': sha(__file__),
              'protocol_sha256': sha(OUT/'AS-C37_protocol.md'), 'pinned_inputs': pin,
              'method_go': False, 'encoder_updates': 0, 'test_loaded': False,
              'entries': [], 'summaries': {}, 'source_verified_row_n': 0}
    dump(path, result)
    try:
        for name, digest in old['source_hashes'].items():
            assert sha(ROOT/name) == digest
        lookup = {(e['split'], e['index']): e for e in old['entries']}
        for split in ('train', 'dev'):
            manifest = ROOT/f'artifacts/manifests/ph_{split}.jsonl'
            assert sha(manifest) == old['splits'][split]['manifest_sha256']
            data = CiCoFeatureDataset(manifest, feature_len=64, alpha=.9, split=split)
            assert len(data) == old['splits'][split]['n']
            for i in range(len(data)):
                if time.time()-started > 300:
                    raise TimeoutError('AS-C37 timeout300s')
                item, prior, record = data[i], lookup[(split, i)], data.records[i]
                h, valid, index = item['h'], item['valid'], item['dense_index']
                assert h.dtype == torch.float32 and h.shape == (64, 1024)
                assert valid.any() and torch.isfinite(h.half()).all()
                assert item['pair_id'] == prior['pair_id'] and item['video_id'] == prior['video_id']
                assert item['view_hash'] == prior['view_hash'] and int(valid.sum()) == prior['valid_n']
                assert record.dense_length == prior['dense_length']
                for field in ('aware', 'agnostic'):
                    source = getattr(record, f'feature_{field}')
                    assert str(source) == prior[f'{field}_path']
                    assert sha(source) == prior[f'{field}_file_sha256']
                assert input_digest(h, valid) == prior['fp32_input_sha256']
                assert input_digest(h.half(), valid) == prior['fp16_input_sha256']
                aware = agnostic = None
                if slot_groups(h, valid) or slot_groups(h.half(), valid):
                    aware = load_dense_feature(record.feature_aware)
                    agnostic = load_dense_feature(record.feature_agnostic)
                    assert torch.isfinite(aware).all() and torch.isfinite(agnostic).all()
                    fused = (1-.9)*aware+.9*agnostic
                    assert torch.equal(fused[index[valid]], h[valid])
                    result['source_verified_row_n'] += 1
                entry = {'split': split, 'index': i, 'pair_id': item['pair_id'],
                         'dense_length': record.dense_length,
                         'valid_dense_indices': index[valid].tolist(),
                         'fp32': describe(h, valid, index, aware, agnostic),
                         'fp16': describe(h.half(), valid, index, aware, agnostic)}
                result['entries'].append(entry)
                if (i+1) % 1000 == 0:
                    print(json.dumps({'split': split, 'done': i+1, 'wall_seconds': time.time()-started}), flush=True)
            result['summaries'][split] = {p: aggregate(result['entries'], split, p) for p in ('fp32', 'fp16')}
            dump(path, result)
        result.update(status='completed', wall_seconds=time.time()-started,
                      source_file_rehash_n=2*len(result['entries']))
        dump(path, result)
        print(json.dumps({k: v for k, v in result.items() if k != 'entries'}, indent=2), flush=True)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
