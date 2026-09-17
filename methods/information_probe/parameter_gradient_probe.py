"""Shared video-parameter VJPs; no optimizer update and no loss redefinition."""
from collections import Counter
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import encode_cico_text
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
from .common import ART, ROOT, dump, rows, sha
from .gradient_probe import gradient_summary
from .scoring import channels
from .text_augmentation_probe import captions, encode

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'
SCALE = 4096.


def flat_grad(grads, params, scale=SCALE):
    values = []
    for g, p in zip(grads, params):
        v = torch.zeros(p.numel()) if g is None else g.detach().cpu().float().reshape(-1)/scale
        assert torch.isfinite(v).all(), 'nonfinite parameter gradient'
        values.append(v)
    return torch.cat(values)


def dot(a, b):
    value = 0.
    for start in range(0, a.numel(), 1000000):
        value += float(torch.dot(a[start:start+1000000].double(), b[start:start+1000000].double()))
    return value


def cosine(a, b):
    return dot(a, b)/max((dot(a, a)*dot(b, b))**.5, 1e-30)


def partition_summary(total, part):
    other = total-part
    nt, npart, no = dot(total, total), dot(part, part), dot(other, other)
    cross = dot(part, other)
    assert abs(nt-(npart+no+2*cross)) <= 1e-5*max(nt, npart+no, 1e-30)
    return {'total_norm': nt**.5, 'part_norm': npart**.5, 'other_norm': no**.5,
            'part_other_cosine': cross/max((npart*no)**.5, 1e-30),
            'part_signed_projection_onto_total': dot(part, total)/max(nt, 1e-30),
            'part_fraction_of_separate_squared_norms': npart/max(npart+no, 1e-30),
            'cross_term': 2*cross,
            'interpretation': 'Pathway decomposition, not additive causal share of an optimizer update.'}


def directional_losses(v, text, aug, vm, tm, am, scale):
    a, b = channels(v, text, vm, tm, scale)
    if aug is not None:
        _, b = channels(v, aug, vm, am, scale)
    target = torch.arange(len(v), device=v.device)
    return ((F.cross_entropy(a, target)+F.cross_entropy(b, target))/2,
            (F.cross_entropy(a.T, target)+F.cross_entropy(b.T, target))/2)


def materialize(dataset, indexes):
    parts = list(DataLoader(Subset(dataset, list(map(int, indexes))), batch_size=128, num_workers=4))
    return torch.cat([b['h'] for b in parts]), torch.cat([b['valid'] for b in parts]).bool()


def vjp_blocks(bridge, dataset, cache, indexes, covectors, params):
    """Keep full128 row shape; separately replay the historical final56 rows."""
    final_start = len(dataset)//128*128
    regular = np.flatnonzero(indexes < final_start)
    special = np.flatnonzero(indexes >= final_start)
    blocks = []
    for start in range(0, len(regular), 128):
        local = regular[start:start+128]
        global_ids = indexes[local]
        padded = np.concatenate((global_ids, np.repeat(global_ids[0], 128-len(local))))
        blocks.append((padded, local, np.arange(len(local))))
    if len(special):
        blocks.append((np.arange(final_start, len(dataset)), special, indexes[special]-final_start))
    accum = {name: torch.zeros(sum(p.numel() for p in params)) for name in covectors}
    unused = set()
    for block_index, (global_ids, local, positions) in enumerate(blocks):
        h, valid = materialize(dataset, global_ids)
        output = bridge.encode_video(h.cuda(), valid.cuda())
        assert torch.equal(output.tokens.detach().cpu(), cache['video_tokens'][global_ids]), 'encoder/cache parity'
        assert torch.equal(output.mask.cpu(), cache['video_mask'][global_ids])
        names = list(covectors)
        for k, name in enumerate(names):
            covector = torch.zeros_like(output.tokens)
            covector[positions] = covectors[name][local].to(covector)*SCALE
            grads = torch.autograd.grad(output.tokens, params, grad_outputs=covector,
                                        retain_graph=k < len(names)-1, allow_unused=True)
            unused.update(i for i, g in enumerate(grads) if g is None)
            accum[name].add_(flat_grad(grads, params))
            del grads, covector
        del output
        print(json.dumps({'vjp_block': block_index+1, 'of': len(blocks)}), flush=True)
    return accum, unused


def smoke(bridge, dataset, cache, params):
    h, valid = materialize(dataset, np.arange(128))
    output = bridge.encode_video(h.cuda(), valid.cuda())
    assert torch.equal(output.tokens.detach().cpu(), cache['video_tokens'][:128])
    args = (cache['text_tokens'][:8].cuda(), None, output.mask[:8], cache['text_mask'][:8].cuda(), None, cache['logit_scale'])
    direct = sum(directional_losses(output.tokens[:8], *args))/2
    gd = flat_grad(torch.autograd.grad(direct*SCALE, params, retain_graph=True, allow_unused=True), params)
    detached = output.tokens[:8].detach().clone().requires_grad_(True)
    indirect = sum(directional_losses(detached, *args))/2
    token_grad, = torch.autograd.grad(indirect, detached)
    covector = torch.zeros_like(output.tokens)
    covector[:8] = token_grad*SCALE
    gi = flat_grad(torch.autograd.grad(output.tokens, params, grad_outputs=covector, allow_unused=True), params)
    error = (dot(gd-gi, gd-gi)/max(dot(gd, gd), 1e-30))**.5
    cos = cosine(gd, gi)
    assert dot(gd, gd) > 0 and error <= 1e-3 and cos >= .99999
    return {'direct_loss': float(direct.detach()), 'chain_loss': float(indirect.detach()),
            'gradient_norm': dot(gd, gd)**.5, 'relative_l2_error': error, 'cosine': cos,
            'encoder_token_parity': True, 'active_examples': 8, 'encoder_rows': 128}


def main():
    path = OUT/'AS-C18-PARAMETER_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    start = time.time()
    report = {'experiment_id': 'AS-C18-PARAMETER', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C18_protocol.md'),
              'updates': 0, 'method_go': False, 'loss_scale': SCALE, 'conditions': [],
              'parameter_scope': 'core.clip.visual only; text/logit-scale fixed; native parameter dtypes',
              'partition_scope': 'video-row/invalid-slot Jacobian paths, NOT duplicate-query losses'}
    dump(path, report)
    try:
        cache = torch.load(ART/'frozen_train.pt', weights_only=True)
        records = rows('train')
        assert cache['ids'] == [r['pair_id'] for r in records]
        assert cache['manifest_sha256'] == sha(ROOT/'artifacts/manifests/ph_train.jsonl')
        checkpoint = ROOT/'runs/ph_base_b512_s42/checkpoints/best_dev.pt'
        assert cache['checkpoint_sha256'] == sha(checkpoint)
        report.update(checkpoint_sha256=sha(checkpoint), cache_sha256=sha(ART/'frozen_train.pt'),
                      manifest_sha256=cache['manifest_sha256'])
        prior = json.loads((OUT/'AS-C12-GRADIENT_run.json').read_text())
        assert prior['status'] == 'completed'
        report['prior_run_sha256'] = sha(OUT/'AS-C12-GRADIENT_run.json')
        config = yaml.safe_load((checkpoint.parents[1]/'resolved_config.yaml').read_text())
        cico_root = ROOT/config['upstream']['cico_root']; sys.path.insert(0, str(cico_root))
        raw = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        state = {k.removeprefix('core.'): v for k, v in raw['model'].items() if k.startswith('core.')}
        core = _load_cico_core_from_state(config, state, cico_root=cico_root, device='cuda')
        core.eval().requires_grad_(False)
        core.clip.visual.requires_grad_(True)
        bridge, tok = CiCoBridge(core), load_cico_tokenizer(config)
        del raw, state
        named = list(core.clip.visual.named_parameters()); params = [p for _, p in named]
        report['parameters'] = [{'name': n, 'count': p.numel(), 'dtype': str(p.dtype)} for n, p in named]
        report['gradient_parameter_n'] = sum(p.numel() for p in params)
        dataset = CiCoFeatureDataset(ROOT/'artifacts/manifests/ph_train.jsonl', feature_len=64, alpha=.9, split='train')
        report['smoke'] = smoke(bridge, dataset, cache, params)
        dump(path, report); print(json.dumps({'smoke': report['smoke']}), flush=True)
        keys = [tuple(encode_cico_text(r['caption_model'], tok, 32)[0].tolist()) for r in records]
        for seed in (42, 1337, 2026):
            aug = encode(bridge, tok, captions(records, seed))
            aug_tokens, aug_mask = aug.tokens.clone(), aug.mask.clone()
            del aug
            indexes = np.random.default_rng(seed).permutation(len(records))[:512]
            counts = Counter(keys[i] for i in indexes)
            duplicate = torch.tensor([counts[keys[i]] > 1 for i in indexes], device='cuda')
            vm, tm = cache['video_mask'][indexes].cuda(), cache['text_mask'][indexes].cuda()
            text = cache['text_tokens'][indexes].cuda()
            for condition in ('clean', 'deployed_aug'):
                v = cache['video_tokens'][indexes].cuda().requires_grad_(True)
                av = None if condition == 'clean' else aug_tokens[indexes]
                am = None if condition == 'clean' else aug_mask[indexes]
                lv, lt = directional_losses(v, text, av, vm, tm, am, cache['logit_scale'])
                gv, = torch.autograd.grad(lv, v, retain_graph=True)
                gt, = torch.autograd.grad(lt, v)
                interface = gradient_summary(gv, gt, vm == 0, duplicate)
                old = next(x for x in prior['batches'] if x['seed'] == seed and x['batch'] == 0 and x['condition'] == condition)
                assert old['indexes'] == indexes.tolist()
                assert abs(float(lv.detach())-old['V2T_loss']) <= 1e-7
                assert abs(float(lt.detach())-old['T2V_loss']) <= 1e-7
                assert abs(interface['global_cosine']-old['global_cosine']) <= 1e-6
                both = (gv+gt)/2
                covectors = {'V': gv.detach().cpu(), 'T': gt.detach().cpu(),
                             'duplicate': (both*duplicate[:, None, None]).detach().cpu(),
                             'invalid': (both*(vm != 0)[..., None]).detach().cpu()}
                interface_all_cos = cosine(gv.detach().cpu().flatten(), gt.detach().cpu().flatten())
                losses = {'V2T': float(lv.detach()), 'T2V': float(lt.detach())}
                del gv, gt, both, v, lv, lt
                torch.cuda.empty_cache()
                gradients, unused = vjp_blocks(bridge, dataset, cache, indexes, covectors, params)
                total = (gradients['V']+gradients['T'])/2
                entry = {'seed': seed, 'condition': condition, 'indexes': indexes.tolist(), 'losses': losses,
                         'prior_valid_interface_parity': True, 'valid_interface': interface,
                         'all_slot_interface_cosine': interface_all_cos,
                         'parameter_directional_cosine': cosine(gradients['V'], gradients['T']),
                         'parameter_V_norm': dot(gradients['V'], gradients['V'])**.5,
                         'parameter_T_norm': dot(gradients['T'], gradients['T'])**.5,
                         'duplicate_video_pathways': partition_summary(total, gradients['duplicate']),
                         'invalid_slot_pathways': partition_summary(total, gradients['invalid']),
                         'unused_parameter_names': [named[i][0] for i in sorted(unused)]}
                report['conditions'].append(entry)
                dump(path, report)
                print(json.dumps({k: v for k, v in entry.items() if k not in ('indexes', 'valid_interface')}), flush=True)
                del gradients, total, covectors
            del aug_tokens, aug_mask
        report.update(status='completed', wall_seconds=time.time()-start,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      limits=['One fixed checkpoint, eval-mode Jacobian, six train conditions only.',
                              'Native FP16 derivatives with fixed loss scaling are not exact real-arithmetic gradients.',
                              'Parameter gradients precede optimizer preconditioning; no actual update or dev effect measured.',
                              'Row pathway attribution is not duplicate-example loss attribution or proof of harm.'])
        dump(path, report)
    except Exception:
        report.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-start)
        dump(path, report)
        raise


if __name__ == '__main__':
    main()
