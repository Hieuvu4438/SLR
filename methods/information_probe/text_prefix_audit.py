"""AS-C31 causal-prefix representation and additive score-credit audit, no fitting."""
from collections import defaultdict
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from slr_common.data.tokenize import encode_cico_text
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
from .common import ART, ROOT, dump, persistent, ranks, rows, sha
from .scoring import channels
from .text_augmentation_probe import OUT, encode


def lcp(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def prefix_inventory(sequences, tokens):
    groups = defaultdict(list)
    for i, seq in enumerate(sequences):
        for k in range(1, len(seq)+1):
            groups[seq[:k]].append(i)
    shared_rows, shared_slots, nonexact_groups, delta, kept = set(), 0, 0, 0., []
    for prefix, indexes in groups.items():
        if len({sequences[i] for i in indexes}) <= 1:
            continue
        pos = len(prefix)  # content prefix length equals last position with SOT at0
        selected = tokens[indexes, pos]
        difference = float((selected-selected[:1]).abs().max())
        delta = max(delta, difference)
        nonexact_groups += int(difference != 0)
        shared_slots += len(indexes)
        shared_rows.update(indexes)
        kept.append({'prefix_ids': list(prefix), 'row_n': len(indexes),
                     'distinct_full_inputs': len({sequences[i] for i in indexes}),
                     'token_max_abs_difference': difference})
    return {'rows': len(sequences), 'content_token_slots': sum(map(len, sequences)),
            'shared_prefix_groups': len(kept), 'shared_prefix_rows': len(shared_rows),
            'shared_prefix_token_slots': shared_slots, 'nonexact_shared_groups': nonexact_groups,
            'max_abs_shared_token_difference': delta,
            'largest_groups': sorted(kept, key=lambda x: (-x['row_n'], x['prefix_ids']))[:10]}


def token_credits(v, t, vm, tm, scale):
    """Exact additive credit decomposition, NOT a causal attribution operator."""
    a = torch.einsum('nfd,nld->nfl', F.normalize(v, dim=-1), F.normalize(t, dim=-1))
    vv, tv = vm == 0, tm == 1
    ac = (a*(a/.07).softmax(-1)*vv[:, :, None]).sum(1)/vv.sum(1)[:, None]
    bc = (a*(a/.07).softmax(-2)).sum(1)*tv/tv.sum(1)[:, None]
    return torch.stack((scale*ac, scale*bc), 1)


def bucket_credits(credit, content_length, common):
    assert 0 <= common <= content_length and content_length+2 <= len(credit)
    return np.array([credit[0], credit[1:1+common].sum(),
                     credit[1+common:1+content_length].sum(), credit[1+content_length],
                     credit[2+content_length:].sum()], dtype=np.float64)


@torch.inference_mode()
def main():
    path = OUT/'AS-C31-PREFIX_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    started = time.time()
    result = {'experiment_id': 'AS-C31-PREFIX', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C31_protocol.md'),
              'method_go': False, 'updates': 0, 'test_loaded': False, 'splits': {}}
    dump(path, result)
    try:
        cp = ROOT/'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
        assert sha(cp) == 'ee45bd2dcad57ff4237eaeb2fc326d288698400c13aac0e51d9df01496b36351'
        config = yaml.safe_load((cp.parents[1]/'resolved_config.yaml').read_text())
        cico_root = ROOT/config['upstream']['cico_root']
        sys.path.insert(0, str(cico_root))
        saved = torch.load(cp, weights_only=True, map_location='cpu', mmap=True)
        state = {k.removeprefix('core.'): v for k, v in saved['model'].items() if k.startswith('core.')}
        core = _load_cico_core_from_state(config, state, cico_root=cico_root, device='cuda')
        del saved, state
        core.eval().requires_grad_(False)
        bridge, tok = CiCoBridge(core), load_cico_tokenizer(config)
        mask = core.clip.build_attention_mask(32)
        expected = torch.full((32, 32), -torch.inf).triu(1)
        assert torch.equal(mask, expected)
        result.update(checkpoint_sha256=sha(cp), causal_mask_exact=True,
                      module_clip_sha256=sha(cico_root/'modules/module_clip.py'))
        for split in ('train', 'dev'):
            if time.time()-started > 300:
                raise TimeoutError('AS-C31 timeout300s')
            rec = rows(split)
            cache = torch.load(ART/f'frozen_{split}.pt', weights_only=True)
            assert cache['checkpoint_sha256'] == result['checkpoint_sha256']
            assert cache['manifest_sha256'] == sha(ROOT/f'artifacts/manifests/ph_{split}.jsonl')
            assert cache['ids'] == [r['pair_id'] for r in rec]
            triples = [encode_cico_text(r['caption_model'], tok, 32) for r in rec]
            sequences = [tuple(x[0][1:int(x[2].sum())-1].tolist()) for x in triples]
            clean = encode(bridge, tok, [r['caption_model'] for r in rec])
            assert torch.equal(clean.tokens.cpu(), cache['text_tokens'])
            assert torch.equal(clean.mask.cpu(), cache['text_mask'])
            inv = prefix_inventory(sequences, cache['text_tokens'])
            inv.update(cache_sha256=sha(ART/f'frozen_{split}.pt'),
                       manifest_sha256=cache['manifest_sha256'], encoder_cache_exact=True)
            result['splits'][split] = inv
            dump(path, result)
            print(json.dumps({'split': split, **{k: v for k, v in inv.items() if k != 'largest_groups'}}), flush=True)
            if split == 'train':
                del cache, clean
                continue
        n = len(sequences)
        base = np.load(ART/'baseline_dev_scores.npy')
        base_ch = np.load(ART/'baseline_dev_channels.npy')
        v, t = cache['video_tokens'].cuda(), cache['text_tokens'].cuda()
        vm, tm = cache['video_mask'].cuda(), cache['text_mask'].cuda()
        replay = np.empty_like(base_ch)
        for i in range(0, n, 64):
            for j in range(0, n, 64):
                replay[:, i:i+64, j:j+64] = torch.stack(channels(v[i:i+64], t[j:j+64], vm[i:i+64], tm[j:j+64], cache['logit_scale'])).cpu().numpy()
        delta = float(np.max(np.abs(replay-base_ch)))
        assert delta <= 2e-5
        assert all(np.array_equal(ranks(replay.mean(0))[d], ranks(base)[d]) for d in ('T2V', 'V2T'))
        result['baseline_replay'] = {'max_channel_delta': delta, 'full_gallery_ranks_exact': True}
        diff = base.copy()
        np.fill_diagonal(diff, -np.inf)
        hard = {'T2V': diff.argmax(0), 'V2T': diff.argmax(1)}
        pairs = sorted({(i, i) for i in range(n)} | {(int(j), i) for i, j in enumerate(hard['T2V'])}
                       | {(i, int(j)) for i, j in enumerate(hard['V2T'])})
        credits, pair_delta = {}, 0.
        for start in range(0, len(pairs), 128):
            chunk = pairs[start:start+128]
            vi, ti = [x[0] for x in chunk], [x[1] for x in chunk]
            value = token_credits(v[vi], t[ti], vm[vi], tm[ti], cache['logit_scale']).cpu().numpy()
            expected = np.stack([base_ch[:, i, j] for i, j in chunk])
            pair_delta = max(pair_delta, float(np.abs(value.sum(-1)-expected).max()))
            credits.update(zip(chunk, value.mean(1).astype(np.float64)))
        assert pair_delta <= 5e-5
        result['pair_channel_credit_max_error'] = pair_delta
        result['credit_bucket_order'] = ['SOT', 'common_content_prefix', 'remaining_content', 'EOT', 'padding']
        populations = persistent()
        result['directions'] = {}
        for direction in ('T2V', 'V2T'):
            records = []
            for i, j0 in enumerate(hard[direction]):
                j = int(j0)
                k = lcp(sequences[i], sequences[j])
                same = sequences[i] == sequences[j]
                negative = (j, i) if direction == 'T2V' else (i, j)
                ti = negative[1]
                pos = bucket_credits(credits[i, i], len(sequences[i]), k)
                neg = bucket_credits(credits[negative], len(sequences[ti]), k)
                buckets = pos-neg
                margin = float(base[i, i]-base[negative])
                assert abs(buckets.sum()-margin) <= 1e-4
                others = [lcp(sequences[i], s) for z, s in enumerate(sequences) if z != i and s != sequences[i]]
                burden = (not same and k >= 3 and margin < -1e-4 and buckets[1] < -1e-4
                          and buckets.sum()-buckets[1] > 1e-4)
                records.append({'query': i, 'confuser': j, 'persistent': bool(populations[direction][i]),
                                'same_deployed_text': same, 'common_content_prefix_n': k,
                                'full_gallery_distinct_text_ge3_fraction': float(np.mean(np.array(others) >= 3)),
                                'margin': margin, 'margin_buckets': buckets.tolist(), 'prefix_burden': bool(burden)})
            p = [x for x in records if x['persistent']]
            summary = {'persistent_n': len(p), 'persistent_same_input_n': sum(x['same_deployed_text'] for x in p),
                       'persistent_distinct_prefix_ge3_n': sum(not x['same_deployed_text'] and x['common_content_prefix_n'] >= 3 for x in p),
                       'persistent_prefix_burden_n': sum(x['prefix_burden'] for x in p),
                       'all_queries_prefix_burden_n': sum(x['prefix_burden'] for x in records),
                       'persistent_gallery_ge3_fraction_mean': float(np.mean([x['full_gallery_distinct_text_ge3_fraction'] for x in p]))}
            summary['broad_prefix_burden_gate'] = summary['persistent_prefix_burden_n'] >= .2*len(p)
            result['directions'][direction] = {'summary': summary, 'queries': records}
            print(json.dumps({'direction': direction, **summary}), flush=True)
        result['broad_signature_gate'] = all(x['summary']['broad_prefix_burden_gate'] for x in result['directions'].values())
        result.update(status='completed', wall_seconds=time.time()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(path, result)
        print(json.dumps({'status': result['status'], 'broad_signature_gate': result['broad_signature_gate'], 'wall_seconds': result['wall_seconds']}), flush=True)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
