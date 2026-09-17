"""AS-C30 official metric checks, final checkpoint replay, and fixed contrasts."""
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
from .clean_train_calibration import adequacy, evaluate
from .common import ART, ROOT, dump, rows, sha
from .freeze_training_comparison import FROZEN, SOURCE, SOURCE_SHA


def main():
    path = OUT/'AS-C30-VALIDATION_run.json'
    if path.exists():
        raise FileExistsError(path)
    started = time.time()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {'status': 'running', 'pid': os.getpid(), 'code_sha256': sha(__file__),
              'dev_or_test_loaded': False, 'method_go': False, 'metric_checks': []}
    dump(path, result)
    try:
        runs = {a: json.loads((OUT/f'AS-C30-{a.upper()}_run.json').read_text()) for a in ('control', 'freeze')}
        assert sha(SOURCE) == SOURCE_SHA
        original_path = OUT/'AS-C20-TRAIN_run.json'
        assert sha(original_path) == '2bcccf1ec19dce738c42ba3a0a0d4d9a1897637dce9ece2d103ff7476a8ec3ad'
        original_run = json.loads(original_path.read_text())
        assert sha(original_run['checkpoint']['path']) == '3d1a135f8a6b93c69989acb821ea719e268fdf147e15fecbaff05fc8ffd1799f'
        assert runs['control']['comparison']['exact_replay']
        assert runs['freeze']['freeze_audit']['frozen_parameter_n'] == 25336320
        assert set(runs['freeze']['freeze_audit']['frozen_keys']) == set(FROZEN)
        partition_path = OUT/'AS-C19-TRAIN-partition.json'
        partition = json.loads(partition_path.read_text())
        records = rows('train')
        assert sha(ROOT/'artifacts/manifests/ph_train.jsonl') == partition['manifest_sha256']
        config_path = ROOT/'runs/ph_base_b512_s42/resolved_config.yaml'
        config = yaml.safe_load(config_path.read_text())
        tok = load_cico_tokenizer(config)
        keys = [tuple(encode_cico_text(r['caption_model'], tok, 32)[0].tolist()) for r in records]
        for arm, run in runs.items():
            assert run['status'] == 'completed' and run['updates'] == 1000
            assert sha(run['checkpoint']['path']) == run['checkpoint']['sha256']
            assert run['partition_sha256'] == sha(partition_path)
            assert run['protocol_sha256'] == sha(OUT/'AS-C30_protocol.md')
            assert run['harness_sha256'] == sha(ROOT/'methods/information_probe/freeze_training_comparison.py')
            assert run['architecture_config_sha256'] == sha(config_path)
            assert run['initialization_code_sha256'] == sha(ROOT/'methods/information_probe/clean_initialization_audit.py')
            assert run['adequacy'] == adequacy(run['evaluations'][0], run['evaluations'][-1])
            for entry in run['evaluations']:
                for split, indexes in [('fit', run['fit_eval_indexes']), ('held', partition['held_indexes'])]:
                    score_path = ART/f'AS-C30-{arm.upper()}/{split}_step{entry["step"]}.npy'
                    assert sha(score_path) == entry[split]['score_sha256']
                    ids = [records[i]['pair_id'] for i in indexes]
                    positives = {i: [i] for i in ids}
                    m = evaluate_score_matrix(np.load(score_path), video_ids=ids, text_ids=ids,
                                              video_to_text=positives, text_to_video=positives)
                    for d in ('T2V', 'V2T'):
                        assert all(m[d][f'R{k}'] == entry[split][d][f'R{k}'] for k in (1, 5, 10))
                    result['metric_checks'].append({'arm': arm, 'step': entry['step'], 'split': split,
                                                    'all_R1_R5_R10_exact': True})
        run = runs['freeze']
        generic_path = ROOT/'artifacts/pretrained/ViT-B-32.pt'
        assert sha(generic_path) == run['generic_clip_sha256']
        generic = torch.jit.load(str(generic_path), map_location='cpu').state_dict()
        core, _ = initialize(config, generic, 42)
        del generic
        core.float()
        initial = {k: core.state_dict()[k].clone() for k in FROZEN}
        cp = run['checkpoint']['path']
        assert sha(cp) == run['checkpoint']['sha256']
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        assert saved['partition_sha256'] == run['partition_sha256'] and saved['step'] == 1000
        assert saved['protocol_sha256'] == run['protocol_sha256']
        assert all(torch.equal(saved['state_dict'][k], v) for k, v in initial.items())
        core.load_state_dict(saved['state_dict'], strict=True)
        del saved
        core.cuda().eval().requires_grad_(False)
        held = partition['held_indexes']
        dataset = CiCoFeatureDataset(ROOT/'artifacts/manifests/ph_train.jsonl', feature_len=64, alpha=.9, split='train')
        items = {i: dataset[i] for i in held}
        score, _ = evaluate(core, CiCoBridge(core), items, held, CiCoCollator(tok, 32), keys)
        original = np.load(ART/'AS-C30-FREEZE/held_step1000.npy')
        assert np.array_equal(score, original)
        result['freeze_final_held_score_replay_exact'] = True
        result['freeze_tables_equal_initial_independent_reload'] = True
        result['initial_evaluations_equal'] = runs['control']['evaluations'][0] == run['evaluations'][0]
        assert result['initial_evaluations_equal']
        c, f = (runs[a]['evaluations'][-1] for a in ('control', 'freeze'))
        result['final_freeze_minus_control_pp'] = {
            split: {d: {f'R{k}': f[split][d][f'R{k}']-c[split][d][f'R{k}'] for k in (1, 5, 10)}
                    for d in ('T2V', 'V2T')} for split in ('fit', 'held')}
        result['final_held_mean_R1_delta_pp'] = f['held']['mean_R1']-c['held']['mean_R1']
        result['adequacy'] = {a: r['adequacy'] for a, r in runs.items()}
        result['run_sha256'] = {a: sha(OUT/f'AS-C30-{a.upper()}_run.json') for a in runs}
        result.update(status='completed', wall_seconds=time.time()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(path, result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
