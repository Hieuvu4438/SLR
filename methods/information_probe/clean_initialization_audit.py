"""Explicit generic-CLIP mapping and internal TRAIN holdout audit; no fitting."""
from collections import Counter, defaultdict
import hashlib
import importlib
import json
import os
import sys
import time
import traceback

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator, encode_cico_text
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import cico_task_config, load_cico_tokenizer
from .common import ROOT, dump, rows, sha
from .scoring import channels

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'
GENERIC_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
RANDOM_KEYS = {'clip.visual.conv1.weight', 'clip.visual.positional_embedding', 'clip.visual.conv2_trans.weight'}
CONSTANT_KEYS = {'clip.logit_scale_first_softmax', 'clip.logit_scale_sec_softmax'}


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def source_fold(source):
    return int.from_bytes(hashlib.sha256(source.encode()).digest()[:8], 'big') % 5


def component_sizes(sources, text_keys):
    parent = list(range(len(sources)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for values in (sources, text_keys):
        first = {}
        for i, key in enumerate(values):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    return sorted(Counter(find(i) for i in range(len(sources))).values(), reverse=True)


def initialize(config, generic, seed=42):
    torch.manual_seed(seed)
    cico_root = ROOT/config['upstream']['cico_root']
    if str(cico_root) not in sys.path:
        sys.path.insert(0, str(cico_root))
    modeling = importlib.import_module('modules.modeling')
    cross = importlib.import_module('modules.module_cross')
    core = modeling.CLIP4Clip(cross.CrossConfig(str(cico_root/'modules/cross-base/cross_config.json')),
                             dict(generic), cico_task_config(config))
    destination = core.state_dict()
    copied, mismatches, unused = {}, {}, []
    for key, value in generic.items():
        target = 'clip.'+key
        if target not in destination:
            unused.append(key)
        elif destination[target].shape != value.shape:
            mismatches[target] = {'source_shape': list(value.shape), 'target_shape': list(destination[target].shape)}
        else:
            copied[target] = value
    missing = set(destination)-set(copied)
    assert missing == RANDOM_KEYS | CONSTANT_KEYS, missing
    assert all(destination[k].ndim == 0 and float(destination[k]) == 1. for k in CONSTANT_KEYS)
    assert set(mismatches) == {'clip.visual.conv1.weight', 'clip.visual.positional_embedding'}
    assert set(unused) <= {'input_resolution', 'context_length', 'vocab_size'}, unused
    loaded = core.load_state_dict(copied, strict=False)
    assert set(loaded.missing_keys) == RANDOM_KEYS | CONSTANT_KEYS and not loaded.unexpected_keys
    destination = core.state_dict()
    assert all(torch.equal(destination[k], value.to(destination[k])) for k, value in copied.items())
    report = {'seed': seed, 'copied_tensor_n': len(copied),
              'copied_parameter_n': sum(destination[k].numel() for k in copied),
              'random_parameter_n': sum(destination[k].numel() for k in RANDOM_KEYS),
              'constant_initialized': {k: float(destination[k]) for k in sorted(CONSTANT_KEYS)},
              'total_parameter_n': sum(p.numel() for p in core.parameters()),
              'mismatches': mismatches, 'unused_source_metadata': unused,
              'random': {k: {'shape': list(destination[k].shape), 'sha256': tensor_hash(destination[k])} for k in sorted(RANDOM_KEYS)},
              'copied_keys': sorted(copied), 'copied_source_equality': True}
    return core, report


def main():
    path = OUT/'AS-C19-INITIALIZATION-V2_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    started = time.time()
    result = {'experiment_id': 'AS-C19-INITIALIZATION-V2', 'status': 'running', 'pid': os.getpid(),
              'previous_failed_run_sha256': sha(OUT/'AS-C19-INITIALIZATION_run.json'),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C19_protocol.md'),
              'optimizer_updates': 0, 'method_go': False, 'assets': {}, 'initializations': [],
              'retrieval_release_checkpoint_loaded': False, 'dev_or_test_loaded': False}
    dump(path, result)
    try:
        generic_path = ROOT/'artifacts/pretrained/ViT-B-32.pt'
        assert sha(generic_path) == GENERIC_SHA
        generic = torch.jit.load(str(generic_path), map_location='cpu').state_dict()
        result['assets']['generic_clip'] = {'path': str(generic_path), 'sha256': GENERIC_SHA}
        for stream in ('ph_domain_agnostic', 'ph_domain_aware_h2s_transfer_gpu'):
            p = ROOT/f'artifacts/features_reextracted/{stream}/extraction_report_train.json'
            extraction = json.loads(p.read_text())
            assert extraction['status'] == 'complete' and extraction['splits'] == ['train']
            assert extraction['completed'] == 7096 and extraction['failed'] == []
            assert sha(extraction['checkpoint']) == extraction['checkpoint_sha256']
            result['assets'][stream] = {'extraction_report_sha256': sha(p), 'checkpoint': extraction['checkpoint'],
                                        'checkpoint_sha256': extraction['checkpoint_sha256'],
                                        'feature_provenance_limit': 'Extractor lineage checked; individual pickle hashes not reverified here.'}
        config_path = ROOT/'runs/ph_base_b512_s42/resolved_config.yaml'
        config = yaml.safe_load(config_path.read_text())
        result['architecture_config_sha256'] = sha(config_path)
        cico_root = ROOT/config['upstream']['cico_root']
        result['upstream_code_hashes'] = {n: sha(cico_root/'modules'/n) for n in ('modeling.py', 'module_clip.py', 'until_module.py')}
        for seed in (42, 42, 1337):
            model, entry = initialize(config, generic, seed)
            result['initializations'].append(entry)
            del model
        a, b, c = result['initializations']
        assert a == b
        assert all(a['random'][k]['sha256'] != c['random'][k]['sha256'] for k in RANDOM_KEYS)
        records = rows('train')
        tok = load_cico_tokenizer(config)
        keys = [tuple(encode_cico_text(r['caption_model'], tok, 32)[0].tolist()) for r in records]
        sources = [r['video_id'].rsplit('-', 1)[0] for r in records]
        folds = [source_fold(s) for s in sources]
        fit, held = [i for i, f in enumerate(folds) if f != 0], [i for i, f in enumerate(folds) if f == 0]
        fit_sources, held_sources = {sources[i] for i in fit}, {sources[i] for i in held}
        assert not fit_sources&held_sources and len(fit)+len(held) == len(records)
        fit_keys, held_keys = {keys[i] for i in fit}, {keys[i] for i in held}
        components = component_sizes(sources, keys)
        result['partition'] = {'manifest_sha256': sha(ROOT/'artifacts/manifests/ph_train.jsonl'),
            'fold_row_counts': dict(Counter(folds)), 'fit_n': len(fit), 'held_n': len(held),
            'fit_source_n': len(fit_sources), 'held_source_n': len(held_sources), 'source_overlap': 0,
            'exact_text_keys_overlap': len(fit_keys&held_keys),
            'held_rows_with_fit_exact_text': sum(keys[i] in fit_keys for i in held),
            'source_text_component_n': len(components), 'source_text_largest_components': components[:10],
            'source_identity_limit': 'Filename prefix inferred recording identity, not independently verified.'}
        dump(OUT/'AS-C19-TRAIN-partition.json', {'manifest_sha256': result['partition']['manifest_sha256'],
             'purpose': 'Internal TRAIN diagnostic only; official splits/positives unchanged',
             'held_fold': 0, 'fit_indexes': fit, 'held_indexes': held,
             'rows': [{'index': i, 'pair_id': r['pair_id'], 'source': sources[i], 'fold': folds[i]} for i, r in enumerate(records)]})
        core, _ = initialize(config, generic, 42)
        core.float().cuda().eval()
        bridge = CiCoBridge(core)
        dataset = CiCoFeatureDataset(ROOT/'artifacts/manifests/ph_train.jsonl', feature_len=64, alpha=.9, split='train')
        loader = DataLoader(Subset(dataset, fit[:32]), batch_size=32,
                            collate_fn=CiCoCollator(tok, 32, augment=True, seed=42))
        batch = next(iter(loader))
        v = bridge.encode_video(batch['h'].cuda(), batch['valid'].cuda())
        t = bridge.encode_text(*(x.cuda() for x in batch['clean_text']))
        ta = bridge.encode_text(*(x.cuda() for x in batch['aug_text']))
        assert all(torch.isfinite(x).all() for x in (v.tokens, t.tokens, ta.tokens))
        scale = core.clip.logit_scale.exp()
        aa, bb = channels(v.tokens, t.tokens, v.mask, t.mask, scale)
        _, ba = channels(v.tokens, ta.tokens, v.mask, ta.mask, scale)
        target = torch.arange(32, device='cuda')
        clean = sum(F.cross_entropy(x, target) for x in (aa, aa.T, bb, bb.T))/4
        augmented = sum(F.cross_entropy(x, target) for x in (aa, aa.T, ba, ba.T))/4
        assert torch.isfinite(clean) and torch.isfinite(augmented)
        augmented.backward()
        grad_n, squared = 0, 0.
        unused = []
        for name, p in core.named_parameters():
            if p.grad is None:
                unused.append(name)
            else:
                assert torch.isfinite(p.grad).all(), name
                grad_n += p.numel()
                squared += float(p.grad.double().square().sum())
        assert squared > 0
        result['smoke'] = {'fit_indexes': fit[:32], 'batch_size': 32, 'dtype': 'float32',
                           'clean_loss': float(clean.detach()), 'augmented_loss': float(augmented.detach()),
                           'gradient_parameter_n': grad_n, 'gradient_norm': squared**.5,
                           'unused_parameter_names': unused, 'finite_forward_backward': True, 'updates': 0}
        result.update(status='completed', wall_seconds=time.time()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            limits=['No deliberately PH-fitted retrieval initialization, not proof of zero pretraining overlap.',
                    'Same architecture, materially different initialization than R0; not an R0 matched method comparison.',
                    'No heldout learning curve or useful residual supervision measured.',
                    'Generic initialization is infrastructure, not a novel method or AS-C07 candidate revival.'])
        dump(path, result)
        print(json.dumps({k: result[k] for k in ('status', 'partition', 'smoke', 'wall_seconds', 'peak_gpu_bytes')}, indent=2))
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
