"""Fixed clean checkpoint, common519-gallery comparison; no causal claim."""
import hashlib
import json
import os
import time
import traceback

import numpy as np
import torch
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator, encode_cico_text
from slr_common.upstream.cico_bridge import CiCoBridge
from slr_common.upstream.factory import load_cico_tokenizer
from .clean_initialization_audit import OUT, initialize
from .clean_train_calibration import compact_metrics, evaluate
from .common import ART, ROOT, dump, rows, sha


def subset_indexes(ids, salt, n=519):
    return sorted(sorted(range(len(ids)), key=lambda i: hashlib.sha256(f'{salt}:{ids[i]}'.encode()).digest())[:n])


def main():
    path = OUT/'AS-C27-COMMONMODEL_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    result = {'experiment_id': 'AS-C27-COMMONMODEL', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C27_protocol.md'),
              'method_go': False, 'updates': 0, 'test_loaded': False, 'held_subsets': []}
    dump(path, result)
    try:
        run_path = OUT/'AS-C20-TRAIN_run.json'
        run = json.loads(run_path.read_text())
        assert run['status'] == 'completed' and run['updates'] == 1000
        cp = run['checkpoint']['path']
        assert sha(cp) == run['checkpoint']['sha256']
        partition = json.loads((OUT/'AS-C19-TRAIN-partition.json').read_text())
        assert sha(OUT/'AS-C19-TRAIN-partition.json') == run['partition_sha256']
        assert sha(ROOT/'artifacts/manifests/ph_train.jsonl') == partition['manifest_sha256']
        result.update(checkpoint_sha256=run['checkpoint']['sha256'], partition_sha256=run['partition_sha256'],
                      calibration_run_sha256=sha(run_path), dev_manifest_sha256=sha(ROOT/'artifacts/manifests/ph_dev.jsonl'))
        config = yaml.safe_load((ROOT/'runs/ph_base_b512_s42/resolved_config.yaml').read_text())
        tok = load_cico_tokenizer(config)
        train, dev = rows('train'), rows('dev')
        held = [train[i] for i in partition['held_indexes']]
        held_ids = [r['pair_id'] for r in held]
        held_keys = [tuple(encode_cico_text(r['caption_model'], tok, 32)[0].tolist()) for r in held]
        score_path = ART/'AS-C20/held_step1000.npy'
        assert sha(score_path) == run['evaluations'][-1]['held']['score_sha256']
        s = np.load(score_path)
        for salt in range(5):
            ix = subset_indexes(held_ids, salt)
            assert len(ix) == 519 and len(set(ix)) == 519
            m = compact_metrics(s[np.ix_(ix, ix)], [held_keys[i] for i in ix])
            result['held_subsets'].append({'salt': salt, 'primary': salt == 0, 'held_local_indexes': ix, 'metrics': m})
        generic_path = ROOT/'artifacts/pretrained/ViT-B-32.pt'
        assert sha(generic_path) == run['generic_clip_sha256']
        generic = torch.jit.load(str(generic_path), map_location='cpu').state_dict()
        core, _ = initialize(config, generic, 42)
        del generic
        core.float()
        saved = torch.load(cp, weights_only=True, map_location='cpu')
        assert saved['step'] == 1000 and saved['partition_sha256'] == run['partition_sha256']
        core.load_state_dict(saved['state_dict'], strict=True)
        del saved
        core.cuda().eval()
        if time.time()-started > 300:
            raise TimeoutError('AS-C27 timeout300s')
        dataset = CiCoFeatureDataset(ROOT/'artifacts/manifests/ph_dev.jsonl', feature_len=64, alpha=.9, split='dev')
        items = [dataset[i] for i in range(len(dataset))]
        assert [x['pair_id'] for x in items] == [r['pair_id'] for r in dev]
        keys = [tuple(encode_cico_text(r['caption_model'], tok, 32)[0].tolist()) for r in dev]
        score, m = evaluate(core, CiCoBridge(core), items, list(range(519)), CiCoCollator(tok, 32), keys)
        p = ART/'AS-C27-clean_dev_scores.npy'
        np.save(p, score)
        result['dev'] = {'score_sha256': sha(p), 'metrics': m}
        fit_sources = {train[i]['video_id'].rsplit('-', 1)[0] for i in partition['fit_indexes']}
        seen = np.array([r['video_id'].rsplit('-', 1)[0] in fit_sources for r in dev])
        assert seen.sum() == 402
        result['dev_source_slices'] = {name: {'n': int(mask.sum()), 'per_query_R1': {
            d: float(100*np.mean(np.array(m[d]['ranks'])[mask] == 0)) for d in ('T2V', 'V2T')}}
            for name, mask in [('seen_prefix', seen), ('unseen_prefix', ~seen)]}
        result.update(status='completed', wall_seconds=time.time()-started, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(path, result)
        print(json.dumps({'status': result['status'], 'held_subset_mean_R1': [x['metrics']['mean_R1'] for x in result['held_subsets']],
                          'dev_mean_R1': m['mean_R1'], 'dev_source_slices': result['dev_source_slices'],
                          'wall_seconds': result['wall_seconds']}, indent=2))
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
