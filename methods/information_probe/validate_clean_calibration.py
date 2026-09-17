"""Independent metric-kernel check and final held-score replay, not training replication."""
import json
import os
import platform
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


def main():
    path = OUT/'AS-C20-VALIDATION_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    output = {'status': 'running', 'pid': os.getpid(), 'code_sha256': sha(__file__),
              'training_replicated': False, 'dev_or_test_loaded': False, 'metric_checks': [],
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                              'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name()}}
    dump(path, output)
    try:
        run_path = OUT/'AS-C20-TRAIN_run.json'
        run = json.loads(run_path.read_text())
        assert run['status'] == 'completed' and run['updates'] == 1000
        assert sha(ROOT/'methods/information_probe/clean_train_calibration.py') == run['code_sha256']
        assert sha(OUT/'AS-C20_protocol.md') == run['protocol_sha256']
        assert sha(ROOT/'methods/information_probe/clean_initialization_audit.py') == run['initialization_code_sha256']
        output['training_run_sha256'] = sha(run_path)
        partition = json.loads((OUT/'AS-C19-TRAIN-partition.json').read_text())
        assert sha(OUT/'AS-C19-TRAIN-partition.json') == run['partition_sha256']
        records = rows('train')
        config = yaml.safe_load((ROOT/'runs/ph_base_b512_s42/resolved_config.yaml').read_text())
        tok = load_cico_tokenizer(config)
        keys = [tuple(encode_cico_text(r['caption_model'], tok, 32)[0].tolist()) for r in records]
        fit_keys = {keys[i] for i in partition['fit_indexes']}
        seen = np.array([keys[i] in fit_keys for i in partition['held_indexes']])
        assert seen.sum() == 72
        for entry in run['evaluations']:
            for name, indexes in [('fit', run['fit_eval_indexes']), ('held', partition['held_indexes'])]:
                p = ART/f'AS-C20/{name}_step{entry["step"]}.npy'
                assert sha(p) == entry[name]['score_sha256']
                s = np.load(p)
                ids = [records[i]['pair_id'] for i in indexes]
                positives = {i: [i] for i in ids}
                m = evaluate_score_matrix(s, video_ids=ids, text_ids=ids, video_to_text=positives, text_to_video=positives)
                for d in ('T2V', 'V2T'):
                    assert all(m[d][f'R{k}'] == entry[name][d][f'R{k}'] for k in (1, 5, 10))
                output['metric_checks'].append({'step': entry['step'], 'gallery': name, 'all_R1_R5_R10_exact': True,
                    'metrics': {d: {k: v for k, v in m[d].items() if k != 'cols'} for d in ('T2V', 'V2T')}})
                if name == 'held' and entry['step'] == 1000:
                    output['final_held_seen_text_diagnostic'] = {}
                    for d in ('T2V', 'V2T'):
                        # T2V here is one optimistic rank/query, NOT official expanded denominator.
                        ranks = np.array(entry[name][d]['ranks'])
                        output['final_held_seen_text_diagnostic'][d] = {
                            'seen_fit_exact_text_n': int(seen.sum()), 'unseen_fit_exact_text_n': int((~seen).sum()),
                            'seen_query_R1': float(100*np.mean(ranks[seen] == 0)),
                            'unseen_query_R1': float(100*np.mean(ranks[~seen] == 0)),
                            'rank_policy': 'optimistic_per_query' if d == 'T2V' else 'official_torch_argsort'}
                del m
        # Reload the final model and raw held TRAIN features; do not rerun optimization.
        cp = run['checkpoint']['path']
        assert sha(cp) == run['checkpoint']['sha256']
        generic_path = ROOT/'artifacts/pretrained/ViT-B-32.pt'
        assert sha(generic_path) == run['generic_clip_sha256']
        generic = torch.jit.load(str(generic_path), map_location='cpu').state_dict()
        core, _ = initialize(config, generic, 42)
        del generic
        core.float()
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        assert saved['step'] == 1000 and saved['partition_sha256'] == run['partition_sha256']
        core.load_state_dict(saved['state_dict'], strict=True)
        del saved
        core.cuda().eval()
        dataset = CiCoFeatureDataset(ROOT/'artifacts/manifests/ph_train.jsonl', feature_len=64, alpha=.9, split='train')
        held = partition['held_indexes']
        items = {i: dataset[i] for i in held}
        score, metrics = evaluate(core, CiCoBridge(core), items, held, CiCoCollator(tok, 32), keys)
        original = np.load(ART/'AS-C20/held_step1000.npy')
        output['held_score_replay'] = {'max_abs_difference': float(np.max(np.abs(score-original))),
                                       'exact_equality': bool(np.array_equal(score, original))}
        assert np.array_equal(score, original), output['held_score_replay']
        output.update(status='completed', wall_seconds=time.time()-started)
        dump(path, output)
        print(json.dumps(output, indent=2))
    except Exception:
        output.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, output)
        raise


if __name__ == '__main__':
    main()
