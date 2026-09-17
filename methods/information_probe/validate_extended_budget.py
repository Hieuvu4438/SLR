"""AS-C42 terminal metric/adequacy checks and final-score replay, not training replay."""
import argparse
import json
import os
import time
import traceback

import numpy as np
import torch
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator, encode_cico_text
from slr_common.evaluation.cico_eval import evaluate_score_matrix
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import load_cico_tokenizer
from .clean_initialization_audit import OUT, initialize
from .clean_train_calibration import evaluate
from .common import ART, ROOT, dump, rows, sha
from .freeze_training_comparison import SOURCE, SOURCE_SHA


def independent_gate(initial, final):
    learning = (min(final['fit'][d]['R1'] for d in ('T2V', 'V2T')) >= 80
                and min(final['held'][d]['R1'] for d in ('T2V', 'V2T')) >= 50
                and final['held']['mean_R1']-initial['held']['mean_R1'] >= 5)
    residual = min(final['held'][d]['strict_different_text_error_n'] for d in ('T2V', 'V2T')) >= 50
    return dict(learning_gate=learning, residual_count_gate=residual,
                diagnostic_adequacy=learning and residual, method_go=False)


def main(attempt=1):
    label = 'AS-C42-BUDGET' + ('-attempt2' if attempt == 2 else '')
    path = OUT/('AS-C42-VALIDATION' + ('-attempt2' if attempt == 2 else '') + '_run.json')
    run_path = OUT/(label+'_run.json')
    run = json.loads(run_path.read_text())
    # Refuse a still-running training run before allocating an output or GPU.
    assert run['status'] == 'completed' and run['updates'] == 8800
    if path.exists():
        raise FileExistsError(path)
    started = time.time()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {'status': 'running', 'pid': os.getpid(), 'code_sha256': sha(__file__),
              'training_run_sha256': sha(run_path), 'training_replicated': False,
              'dev_or_test_loaded': False, 'method_go': False, 'metric_checks': []}
    dump(path, result)
    try:
        if attempt == 2:
            recovery = run['recovery']
            assert recovery['prefix_exact'] and recovery['full_interrupted_prefix_checked']
            assert recovery['checked_training_records'] == 264 and recovery['checked_evaluation_records'] == 3
            assert sha(ROOT/'methods/information_probe/recover_extended_budget.py') == recovery['wrapper_sha256']
            assert sha(OUT/'AS-C42-BUDGET_run.json') == recovery['interrupted_run_sha256']
            result['recovery_check'] = recovery
        assert sha(SOURCE) == run['source_sha256'] == run['code_sha256'] == SOURCE_SHA
        assert sha(OUT/'AS-C42_protocol.md') == run['protocol_sha256']
        assert sha(ROOT/'methods/information_probe/extended_budget_calibration.py') == run['harness_sha256']
        assert all(sha(ROOT/p) == h for p, h in run['helper_sha256'].items())
        assert run['initialization_check']['evaluation_exact']
        assert all(all(x.values()) for x in run['initialization_check']['score_checks'].values())
        assert [e['step'] for e in run['evaluations']] == [0, 2200, 4400, 8800]
        assert [e['step'] for e in run['training']] == list(range(25, 8801, 25))
        assert run['training'][-1]['epoch'] == 199
        partition_path = OUT/'AS-C19-TRAIN-partition.json'
        assert sha(partition_path) == run['partition_sha256']
        partition = json.loads(partition_path.read_text())
        rec = rows('train')
        assert sha(ROOT/'artifacts/manifests/ph_train.jsonl') == partition['manifest_sha256']
        config_path = ROOT/'runs/ph_base_b512_s42/resolved_config.yaml'
        assert sha(config_path) == run['architecture_config_sha256']
        config = yaml.safe_load(config_path.read_text())
        tok = load_cico_tokenizer(config)
        keys = [tuple(encode_cico_text(r['caption_model'], tok, 32)[0].tolist()) for r in rec]
        recomputed = []
        for entry in run['evaluations']:
            measured = {'step': entry['step']}
            for split, indexes in [('fit', run['fit_eval_indexes']), ('held', partition['held_indexes'])]:
                score_path = ART/label/f'{split}_step{entry["step"]}.npy'
                assert sha(score_path) == entry[split]['score_sha256']
                s = np.load(score_path)
                assert np.isfinite(s).all() and s.shape == (1375, 1375)
                ids = [rec[i]['pair_id'] for i in indexes]
                positives = {i: [i] for i in ids}
                official = evaluate_score_matrix(s, video_ids=ids, text_ids=ids,
                                                video_to_text=positives, text_to_video=positives)
                local_keys = [keys[i] for i in indexes]
                different = np.array([[a != b for b in local_keys] for a in local_keys])
                rivals = np.where(different, s, -np.inf)
                measured[split] = {}
                for d, axis in (('T2V', 0), ('V2T', 1)):
                    m = {f'R{k}': official[d][f'R{k}'] for k in (1, 5, 10)}
                    assert all(value == entry[split][d][key] for key, value in m.items())
                    errors = np.flatnonzero(rivals.max(axis=axis) > np.diag(s)).tolist()
                    assert errors == entry[split][d]['strict_different_text_error_indexes']
                    m['strict_different_text_error_n'] = len(errors)
                    assert len(errors) == entry[split][d]['strict_different_text_error_n']
                    rr = ((s > np.diag(s)[None, :]).sum(0) if d == 'T2V' else
                          torch.argsort(torch.argsort(torch.from_numpy(s), dim=1, descending=True), dim=1).diagonal().numpy())
                    assert rr.tolist() == entry[split][d]['ranks']
                    measured[split][d] = m
                measured[split]['mean_R1'] = sum(measured[split][d]['R1'] for d in ('T2V', 'V2T'))/2
                assert measured[split]['mean_R1'] == entry[split]['mean_R1']
                result['metric_checks'].append({'step': entry['step'], 'split': split,
                    'official_R1_R5_R10_exact': True, 'residual_indexes_exact': True, 'ranks_exact': True})
            recomputed.append(measured)
        result['adequacy'] = independent_gate(recomputed[0], recomputed[-1])
        assert result['adequacy'] == run['adequacy']
        original_path = OUT/'AS-C20-TRAIN_run.json'
        assert sha(original_path) == run['parent_run_sha256']
        old = json.loads(original_path.read_text())
        result['endpoint_contrast'] = {split: {d: {f'R{k}': recomputed[-1][split][d][f'R{k}']-old['evaluations'][-1][split][d][f'R{k}']
                for k in (1, 5, 10)} for d in ('T2V', 'V2T')} for split in ('fit', 'held')}
        result['fixed_evaluations'] = recomputed
        generic_path = ROOT/'artifacts/pretrained/ViT-B-32.pt'
        assert sha(generic_path) == run['generic_clip_sha256']
        generic = torch.jit.load(str(generic_path), map_location='cpu').state_dict()
        core, _ = initialize(config, generic, 42)
        del generic
        core.float()
        cp = run['checkpoint']['path']
        assert sha(cp) == run['checkpoint']['sha256']
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        assert saved['step'] == 8800 and saved['partition_sha256'] == run['partition_sha256']
        assert saved['protocol_sha256'] == run['protocol_sha256']
        core.load_state_dict(saved['state_dict'], strict=True)
        del saved
        core.cuda().eval().requires_grad_(False)
        held = partition['held_indexes']
        dataset = CiCoFeatureDataset(ROOT/'artifacts/manifests/ph_train.jsonl', feature_len=64, alpha=.9, split='train')
        items = {i: dataset[i] for i in held}
        score, _ = evaluate(core, CiCoBridge(core), items, held, CiCoCollator(tok, 32), keys)
        original_score = np.load(ART/label/'held_step8800.npy')
        result['final_held_score_replay'] = {'exact': bool(np.array_equal(score, original_score)),
                                           'max_abs_error': float(np.abs(score-original_score).max())}
        assert result['final_held_score_replay']['exact']
        result.update(status='completed', wall_seconds=time.time()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(path, result)
        print(json.dumps({k: v for k, v in result.items() if k != 'fixed_evaluations'}, indent=2))
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, choices=(1, 2), default=1)
    main(parser.parse_args().attempt)
