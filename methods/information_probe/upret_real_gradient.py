"""Paired TRAIN-only UPRet reduction gradients; no optimizer or evaluation."""
import argparse
import ast
import hashlib
import inspect
import json
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from method1 import baseline
from method1.config import load_config
from method1.data import _load_feature, augment_caption
from method1.distributed import DistributedRuntime
from method1.model_factory import build_upret_model
from method1.provenance import implementation_source_report
from method1.sampling import mix_and_sample_features
from method1.token_spans import tokenize_with_spans
from method1.upstream import create_upret_tokenizer

from .common import ROOT, dump, rows, sha

OUT = ROOT / 'docs/proposal7/evidence/autonomous_search'


def transport_with_matrix():
    tree = ast.parse(inspect.getsource(baseline.distribution_transport_score))
    fn = tree.body[0]
    assert isinstance(fn.body[-1], ast.Return)
    assert ast.unparse(fn.body[-1].value) == 'output'
    fn.body[-1].value = ast.parse('(output, transported)', mode='eval').body
    scope = dict(vars(baseline))
    exec(compile(ast.fix_missing_locations(tree), '<paired-native-transport>', 'exec'), scope)
    return scope[fn.name]


def score_energy(x):
    return .5 * ((x - x.mean(0, keepdim=True)).square().mean()
                 + (x - x.mean(1, keepdim=True)).square().mean())


def objective(core, v, t, ot):
    scale, beta = core.clip.logit_scale.exp().float(), float(core.dual_mix)
    lv, lt = scale * v + float(core.ot_weight) * ot, scale * t + float(core.ot_weight) * ot
    y = torch.arange(len(v), device=v.device)
    return .5 * (beta * F.cross_entropy(lv, y) + (1-beta) * F.cross_entropy(lv.T, y)
                 + beta * F.cross_entropy(lt.T, y) + (1-beta) * F.cross_entropy(lt, y))


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_group(name):
    if name.startswith('clip.visual.'):
        return 'visual_encoder'
    if name.startswith('clip.') and 'logit_scale' not in name:
        return 'text_encoder'
    return 'distribution' if name.startswith('dist_') else 'score_heads'


def compare_gradients(names, base, other):
    stats = {group: torch.zeros(4, device='cuda', dtype=torch.float64)
             for group in ('visual_encoder', 'text_encoder', 'distribution', 'score_heads')}
    for name, left, right in zip(names, base, other, strict=True):
        if left is None and right is None:
            continue
        left = torch.zeros_like(right) if left is None else left
        right = torch.zeros_like(left) if right is None else right
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        left, right = left.double(), right.double()
        stats[gradient_group(name)] += torch.stack((left.square().sum(), right.square().sum(),
                                                   (left*right).sum(), (left-right).square().sum()))
    result = {}
    for group, values in stats.items():
        a, b, dot, delta = values.cpu().tolist()
        result[group] = {'base_norm': a**.5, 'arm_norm': b**.5,
                         'cosine': dot/(a*b)**.5 if a*b > 0 else None,
                         'relative_delta': (delta/a)**.5 if a > 0 else None}
    return result


def build_batches(config, tokenizer):
    all_rows = rows('train')
    assert len(all_rows) == 7096 and all(r['split'] == 'train' for r in all_rows)
    selected = np.random.default_rng(20260916).permutation(len(all_rows))[:128]
    source = ROOT / config.data.source_annotations['train']
    with source.open('rb') as handle:
        captions = pickle.load(handle)
    batches, evidence = [], []
    for offset in range(0, 128, 32):
        items, selected_rows = [], []
        for idx in selected[offset:offset+32]:
            row = all_rows[int(idx)]
            uid, caption = row['video_id'], row['caption_model']
            # Confirm captions against the native TRAIN source without all-split manifests.
            native = captions[uid]
            if isinstance(native, list):
                assert len(native) == 1
                native = native[0]
            assert native['text'].strip() == caption.strip()
            agnostic = ROOT / config.data.agnostic_root / 'train' / f'{uid}.pkl'
            aware = ROOT / config.data.aware_root / 'train' / f'{uid}.pkl'
            features, valid, _ = mix_and_sample_features(
                _load_feature(str(agnostic)), _load_feature(str(aware)),
                agnostic_weight=config.data.agnostic_weight, feature_len=64)
            text_uid = f'ph:train:text:{uid}'
            augmented = augment_caption(caption, seed=42, epoch=0, text_uid=text_uid)
            original, aug = [tokenize_with_spans(value, text_uid=text_uid,
                             tokenizer=tokenizer, max_positions=32) for value in (caption, augmented)]
            ignore = np.ones(65, dtype=np.bool_)
            ignore[1:] = ~valid
            items.append({'video_features': torch.from_numpy(features.T[:, :, None].copy()),
                          'video_ignore_raw': torch.from_numpy(ignore),
                          'input_ids': torch.tensor(original.input_ids),
                          'text_valid': torch.tensor(original.text_valid, dtype=torch.bool),
                          'input_ids_aug': torch.tensor(aug.input_ids),
                          'text_aug_valid': torch.tensor(aug.text_valid, dtype=torch.bool),
                          'token_type_ids': torch.zeros(32, dtype=torch.long)})
            selected_rows.append({'index': int(idx), 'video_id': uid,
                                  'agnostic_path': str(agnostic), 'aware_path': str(aware),
                                  'agnostic_sha256': sha(agnostic), 'aware_sha256': sha(aware)})
        batches.append({key: torch.stack([item[key] for item in items]).cuda() for key in items[0]})
        evidence.append(selected_rows)
    return batches, evidence, sha(source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    path = OUT / ('UPRET-REAL-GRADIENT_replay.json' if args.replay else 'UPRET-REAL-GRADIENT_run.json')
    if path.exists():
        raise FileExistsError(path)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.55)
    config_path = ROOT / 'methods/sssc/configs/method1/ph_seed42_base_initial.yaml'
    config = load_config(config_path)
    tokenizer = create_upret_tokenizer(ROOT/'third_party/UPRet', config.model.bpe_path)
    batches, batch_evidence, caption_sha = build_batches(config, tokenizer)
    model, initialization = build_upret_model(config)
    model = model.cuda().train()
    pair_transport = transport_with_matrix()
    source_report = implementation_source_report()
    checkpoint_path = ROOT / 'runs/method1/ph/base/seed42/best_dev.pt'
    results = []
    for state in ('clip_initialization', 'partial_step_checkpoint'):
        metadata = {'label': state}
        if state == 'partial_step_checkpoint':
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            assert checkpoint['artifact_hashes']['implementation_source'] == source_report
            model.load_state_dict(checkpoint['student_state_dict'], strict=True)
            metadata.update(step=checkpoint['global_step'], sha256=sha(checkpoint_path),
                            training_complete=bool(checkpoint.get('training_run_complete', False)))
            del checkpoint
        metadata['state_sha256_before'] = state_hash(model)
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        names, parameters = zip(*named)
        check_indexes = [names.index(name) for name in
                         ('clip.visual.conv2_trans.weight', 'clip.visual.proj',
                          'clip.text_projection', 'clip.ln_final.weight')]
        entries = []
        for batch_idx, batch in enumerate(batches):
            torch.manual_seed(20260915 + batch_idx)
            torch.cuda.manual_seed_all(20260915 + batch_idx)
            with torch.set_grad_enabled(batch_idx > 0):
                encoding = baseline.encode_local(model, batch)
                max_score, transported = pair_transport(model, encoding, seed=42,
                                                        optimizer_step=batch_idx, microstep=0)
                sum_score = transported.sum((-2, -1))
                if batch_idx == 0:
                    assert score_energy(sum_score) > 0 and score_energy(max_score) > 0
                    coefficient = float((score_energy(max_score)/score_energy(sum_score)).sqrt())
                    metadata['calibrated_coefficient'] = coefficient
                    print(json.dumps({'state': state, 'calibration': coefficient}), flush=True)
                    continue
                v, t = baseline.directional_scores_dense(
                    model, encoding.video_raw, encoding.video_ignore_raw, encoding.text_raw,
                    encoding.text_valid, text_aug_raw=encoding.text_aug_raw,
                    text_aug_valid=encoding.text_aug_valid,
                    temperature=config.model.inner_similarity_temperature)
                arms = {'max': max_score, 'sum': sum_score,
                        'scaled_sum': coefficient*sum_score, 'no_ot': torch.zeros_like(max_score)}
                losses = {name: objective(model, v, t, ot) for name, ot in arms.items()}
                reference = baseline.baseline_loss_from_local(
                    model, encoding, runtime=DistributedRuntime(), seed=42,
                    optimizer_step=batch_idx, temperature=config.model.inner_similarity_temperature)
                scalar_error = float((reference-losses['max']).detach().abs())
                assert scalar_error <= 1e-6
                ref_grad = torch.autograd.grad(reference, [parameters[i] for i in check_indexes],
                                               retain_graph=True, allow_unused=True)
                base_grad = torch.autograd.grad(losses['max'], parameters, retain_graph=True,
                                                allow_unused=True)
                assert all((g is None) == (base_grad[i] is None) for i, g in zip(check_indexes, ref_grad))
                parity_error = max(float((g-base_grad[i]).abs().max())
                                   for i, g in zip(check_indexes, ref_grad) if g is not None)
                assert parity_error <= 1e-6
                entry = {'batch': batch_idx, 'scalar_parity_error': scalar_error,
                         'selected_parameter_gradient_parity_error': parity_error,
                         'unused_parity_parameters': [names[i] for i, g in zip(check_indexes, ref_grad) if g is None],
                         'logit_scale': float(model.clip.logit_scale.exp().detach()), 'arms': {}}
                for arm, loss in losses.items():
                    gradients = base_grad if arm == 'max' else torch.autograd.grad(
                        loss, parameters, retain_graph=True, allow_unused=True)
                    entry['arms'][arm] = {
                        'loss': float(loss.detach()),
                        'score_energy': float(score_energy(arms[arm]).detach()),
                        'score_min': float(arms[arm].detach().min()),
                        'score_max': float(arms[arm].detach().max()),
                        'gradients': compare_gradients(names, base_grad, gradients)}
                    for group in ('visual_encoder', 'text_encoder'):
                        assert entry['arms'][arm]['gradients'][group]['base_norm'] > 0
                    del gradients
                    print(json.dumps({'state': state, 'batch': batch_idx, 'arm': arm,
                                      'loss': entry['arms'][arm]['loss']}), flush=True)
                entries.append(entry)
                del encoding, max_score, transported, sum_score, v, t, arms, losses, reference, ref_grad, base_grad, loss
        metadata['state_sha256_after'] = state_hash(model)
        assert metadata['state_sha256_before'] == metadata['state_sha256_after']
        metadata['screen_batches'] = entries
        results.append(metadata)
    medians = {group: float(np.median([e['arms']['scaled_sum']['gradients'][group]['relative_delta']
                                     for e in results[-1]['screen_batches']]))
               for group in ('visual_encoder', 'text_encoder')}
    result = {'status': 'completed', 'code_sha256': sha(__file__),
              'protocol_sha256': sha(OUT/'UPRET_real_gradient_protocol.md'),
              'config_sha256': sha(config_path), 'train_manifest_sha256': sha(ROOT/'artifacts/manifests/ph_train.jsonl'),
              'train_caption_sha256': caption_sha, 'implementation': source_report,
              'initialization_checkpoint_sha256': sha(ROOT/config.model.clip_checkpoint_path),
              'batch_evidence': batch_evidence, 'states': results,
              'partial_scaled_sum_median_relative_gradient_delta': medians,
              'sensitivity_lead': all(value >= .01 for value in medians.values()),
              'torch_version': str(torch.__version__), 'cuda_device': torch.cuda.get_device_name(),
              'training_updates': 0, 'dev_evaluated': False, 'test_loaded': False, 'method_go': False,
              'wall_seconds': time.monotonic()-started}
    if args.replay:
        original = json.loads((OUT/'UPRET-REAL-GRADIENT_run.json').read_text())
        assert {k:v for k,v in original.items() if k != 'wall_seconds'} == {
            k:v for k,v in result.items() if k != 'wall_seconds'}
    dump(path, result)
    print(json.dumps({'output': str(path), 'medians': medians, 'lead': result['sensitivity_lead'],
                      'wall_seconds': result['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
