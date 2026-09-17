"""Frozen omission restoration with matched length/positions, no learned method."""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import yaml

from slr_common.data.tokenize import encode_cico_text
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
from .common import ART, ROOT, dump, metrics, ranks, rows, sha
from .scoring import channels
from .text_augmentation_probe import encode
from .text_omission_inventory import OUT, selected_indexes


def repeat_retained(tokens, capacity=30):
    keep = selected_indexes(len(tokens), capacity)
    return [tokens[min(keep, key=lambda k: (abs(k-i), k))] for i in range(len(tokens))] if tokens else []


@torch.inference_mode()
def main():
    path = OUT/'AS-C23-RESTORE_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    started = time.time()
    result = {'experiment_id': 'AS-C23-RESTORE', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C23_protocol.md'),
              'method_go': False, 'training_updates': 0, 'test_loaded': False, 'variants': {}}
    dump(path, result)
    try:
        cp = ROOT/'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
        config = yaml.safe_load((cp.parents[1]/'resolved_config.yaml').read_text())
        cache = torch.load(ART/'frozen_dev.pt', weights_only=True)
        rec = rows('dev')
        assert cache['checkpoint_sha256'] == sha(cp)
        assert cache['manifest_sha256'] == sha(ROOT/'artifacts/manifests/ph_dev.jsonl')
        assert cache['ids'] == [r['pair_id'] for r in rec]
        result.update(checkpoint_sha256=sha(cp), cache_sha256=sha(ART/'frozen_dev.pt'), manifest_sha256=cache['manifest_sha256'])
        cico_root = ROOT/config['upstream']['cico_root']
        sys.path.insert(0, str(cico_root))
        saved = torch.load(cp, weights_only=True, map_location='cpu', mmap=True)
        state = {k.removeprefix('core.'): v for k, v in saved['model'].items() if k.startswith('core.')}
        core = _load_cico_core_from_state(config, state, cico_root=cico_root, device='cuda')
        core.eval().requires_grad_(False)
        bridge, tok = CiCoBridge(core), load_cico_tokenizer(config)
        del saved, state
        texts = [r['caption_model'] for r in rec]
        clean = encode(bridge, tok, texts)
        assert torch.equal(clean.tokens.cpu(), cache['text_tokens']) and torch.equal(clean.mask.cpu(), cache['text_mask'])
        v, vm = cache['video_tokens'].cuda(), cache['video_mask'].cuda()
        base = np.load(ART/'baseline_dev_scores.npy')
        base_ch = np.load(ART/'baseline_dev_channels.npy')
        canonical = np.empty_like(base_ch)
        for i in range(0, 519, 64):
            for j in range(0, 519, 64):
                ab = channels(v[i:i+64], clean.tokens[j:j+64], vm[i:i+64], clean.mask[j:j+64], cache['logit_scale'])
                canonical[:, i:i+64, j:j+64] = torch.stack(ab).cpu().numpy()
        assert np.abs(canonical-base_ch).max() <= 2e-5
        assert all(np.array_equal(ranks(canonical.mean(0))[d], ranks(base)[d]) for d in ('T2V', 'V2T'))
        tokens = [tok.tokenize(t) for t in texts]
        affected = np.array([len(t) > 30 for t in tokens])
        assert affected.sum() == 10 and max(map(len, tokens)) <= 75
        result['affected_indexes'] = np.flatnonzero(affected).tolist()
        result['identity'] = {'tokens_exact': True, 'rank_parity': True, 'max_channel_delta': float(np.abs(canonical-base_ch).max())}
        for name in ('full', 'repeat_retained'):
            seq = [t if name == 'full' else repeat_retained(t) for t in tokens]
            ids = []
            for t in seq:
                x = tok.convert_tokens_to_ids(['<|startoftext|>', *t, '<|endoftext|>'])
                ids.append(x + [0]*(77-len(x)))
            ids = torch.tensor(ids, device='cuda')
            encoded, masks = [], []
            for i in range(0, 519, 128):
                if time.time()-started > 300:
                    raise TimeoutError('AS-C23 hard timeout300s')
                x = ids[i:i+128]
                enc = bridge.encode_text(x, torch.zeros_like(x), torch.ones_like(x))
                encoded.append(enc.tokens)
                masks.append(enc.mask)
            encoded, masks = torch.cat(encoded), torch.cat(masks)
            score = base.copy()
            for j in np.flatnonzero(affected):
                length = len(tokens[j])+2
                assert masks[j, :length].eq(1).all() and masks[j, length:].eq(0).all()
                assert length == int(clean.mask[j].sum())+len(tokens[j])-30
                for i in range(0, 519, 64):
                    a, b = channels(v[i:i+64], encoded[j:j+1, :length], vm[i:i+64], masks[j:j+1, :length], cache['logit_scale'])
                    score[i:i+64, j:j+1] = ((a+b)/2).cpu().numpy()
            assert np.array_equal(score[:, ~affected], base[:, ~affected])
            m = metrics(score, base)
            np.save(ART/f'AS-C23-{name}_scores.npy', score)
            dump(OUT/f'AS-C23-{name}_metrics.json', m)
            result['variants'][name] = {'mean_R1': m['official_mean_R1'],
                'directions': {d: {**{k: m[o][k] for k in ('R1', 'R5', 'R10')},
                                    'persistent_mean_rank_delta': m[d]['persistent_mean_rank_delta'],
                                    'affected_query_ranks': [m[d]['ranks'][j] for j in np.flatnonzero(affected)],
                                    'changed_query_indexes': np.flatnonzero(np.array(m[d]['rank_delta']) != 0).tolist()}
                               for d, o in [('T2V', 'official_T2V'), ('V2T', 'V2T')]}}
        bm = metrics(base, base)
        full, control = result['variants']['full'], result['variants']['repeat_retained']
        result['lead_gate'] = (full['mean_R1']-max(control['mean_R1'], bm['official_mean_R1']) >= .5
            and all(full['directions'][d]['R1']-bm[o]['R1'] >= -.25
                    and all(full['directions'][d][k]-bm[o][k] >= -.5 for k in ('R5', 'R10'))
                    and full['directions'][d]['persistent_mean_rank_delta'] < 0
                    for d, o in [('T2V', 'official_T2V'), ('V2T', 'V2T')]))
        result.update(status='completed', wall_seconds=time.time()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(path, result)
        print(json.dumps(result, indent=2))
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
