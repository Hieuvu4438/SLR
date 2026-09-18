"""Fresh CiCo CSL DEV replay with existing grouped-gallery infrastructure."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import yaml

from extraction_resume import atomic_json
from inventory import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    cli = parser.parse_args()
    out = ROOT/'artifacts/slret_goal'/cli.run_id
    out.mkdir(exist_ok=False)
    start = time.time()
    report = dict(run_id=cli.run_id, status='running', pid=os.getpid(), command=sys.argv,
                  script_sha256=sha(__file__), test_loaded=False, optimizer_updates=0,
                  selection_split='historically_exposed_CSL_dev',
                  commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'])).hexdigest())
    atomic_json(out/'run.json', report)
    try:
        assert shutil.disk_usage(out).free > 15*1024**3
        size = sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())
        assert size+128*1024**2 < 15*1024**3, 'Reserve output under campaign cap'
        torch.set_num_threads(4)
        torch.manual_seed(42)
        np.random.seed(42)
        sys.path[:0] = [str(ROOT/'shared'), str(ROOT/'methods/elsc')]
        from elsc.config import config_hash
        from slr_common.data.cico_dataset import CiCoFeatureDataset
        from slr_common.data.tokenize import CiCoCollator
        from slr_common.evaluation.runtime import encode_gallery, score_gallery_blockwise
        from slr_common.evaluation.cico_eval import evaluate_score_matrix
        from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
        from slr_common.upstream.cico_bridge import CiCoBridge
        from slr_common.utils import ordered_hash
        root = ROOT/'runs/csl_base_b512_s42'
        config_path = root/'resolved_config.yaml'
        cfg = yaml.safe_load(config_path.read_text())
        selection = json.loads((root/'selection.json').read_text())
        assert cfg['method'] == 'baseline' and not cfg['model']['adapter']['enabled']
        assert selection['config_hash'] == config_hash(cfg)
        checkpoint = root/selection['checkpoint']
        manifest = ROOT/cfg['data']['dev_manifest']
        assert sha(checkpoint) == selection['checkpoint_sha256']
        assert sha(manifest) == selection['dev_manifest_sha256']
        data = CiCoFeatureDataset(manifest, feature_len=cfg['data']['feature_len'],
                                  alpha=cfg['data']['alpha'], split='dev')
        assert len(data) == 1077
        feature_hashes = {}
        for r in data.records:
            for kind in ['agnostic', 'aware']:
                path = getattr(r, 'feature_'+kind)
                meta = json.loads(__import__('pathlib').Path(path+'.meta.json').read_text())
                actual = sha(path)
                assert actual == meta['feature_sha256']
                assert meta['checkpoint_sha256'] == cfg['sources']['feature_'+kind+'_checkpoint_sha256']
                assert meta['recipe_sha256'] == cfg['sources']['feature_recipe_sha256']
                feature_hashes[path] = actual
        atomic_json(out/'feature_hashes.json', feature_hashes)
        raw = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        state = {k.removeprefix('core.'):v for k,v in raw['model'].items() if k.startswith('core.')}
        cico_root = ROOT/cfg['upstream']['cico_root']
        sys.path.insert(0, str(cico_root))
        core = _load_cico_core_from_state(cfg, state, cico_root=cico_root, device='cpu')
        assert set(core.state_dict()) == set(state)
        assert all(torch.equal(core.state_dict()[k], v) for k,v in state.items())
        core = core.cuda().eval().requires_grad_(False)
        bridge = CiCoBridge(core)

        class Baseline:
            def eval(self):
                core.eval()

            def encode_video(self, h, valid):
                return bridge.encode_video(h, valid), h

            def encode_text(self, *args):
                return bridge.encode_text(*args)

        loader = torch.utils.data.DataLoader(data, batch_size=128, num_workers=0, shuffle=False,
                    collate_fn=CiCoCollator(load_cico_tokenizer(cfg), cfg['data']['max_words'], augment=False))
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            gallery = encode_gallery(Baseline(), loader, torch.device('cuda'))
            assert len(gallery.video_ids) == 1077 and len(gallery.text_ids) == 797
            scores = score_gallery_blockwise(bridge, gallery, device=torch.device('cuda'),
                                             dual_mix=cfg['model']['dual_mix'], video_block=128, text_block=128)
        old_path = root/'evaluation/dev/scores_video_x_text.npy'
        old = json.loads((root/'evaluation/dev/metrics.json').read_text())
        assert ordered_hash(gallery.video_ids) == old['id_hashes']['videos']
        assert ordered_hash(gallery.text_ids) == old['id_hashes']['texts']
        actual = evaluate_score_matrix(scores, video_ids=gallery.video_ids, text_ids=gallery.text_ids,
                     video_to_text=gallery.video_to_text, text_to_video=gallery.text_to_video)
        delta = float(np.max(np.abs(scores-np.load(old_path))))
        rank_changes = {d:sum(a != b for a,b in zip(actual[d]['cols'],old[d]['cols'])) for d in ['T2V','V2T']}
        np.save(out/'scores_video_x_text.npy', scores)
        atomic_json(out/'gallery.json', dict(video_ids=gallery.video_ids, text_ids=gallery.text_ids,
                    video_to_text=gallery.video_to_text, text_to_video=gallery.text_to_video))
        # Full score matrix + ordered IDs retain all rankings without huge redundant JSON.
        actual.pop('per_query')
        atomic_json(out/'metrics.json', actual)
        report.update(config_sha256=sha(config_path), checkpoint_sha256=sha(checkpoint),
                      manifest_sha256=sha(manifest), feature_files_verified=len(feature_hashes),
                      checkpoint_tensor_parity='exact', gallery=actual['gallery'],
                      metrics={d:{k:v for k,v in actual[d].items() if k!='cols'} for d in ['T2V','V2T']},
                      metric_kernel=actual['metric_kernel'], max_absolute_score_delta=delta,
                      historical_rank_changes=rank_changes, historical_score_sha256=sha(old_path),
                      score_sha256=sha(out/'scores_video_x_text.npy'),
                      source_sha256={str(p):sha(p) for p in [cico_root/'modules/modeling.py',
                          ROOT/'shared/slr_common/evaluation/runtime.py',ROOT/'shared/slr_common/evaluation/cico_eval.py']},
                      hardware=torch.cuda.get_device_name(), torch=torch.__version__,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      feature_regime='BSL5K_plus_H2S_transfer_aware_not_target_domain_release')
        assert delta <= 1e-4 and not any(rank_changes.values()), 'Historical parity failed'
        report.update(status='completed', exit_status=0,
                      decision='Fresh grouped CSL baseline replay; historically selected DEV is not independent confirmation.')
    except Exception:
        report.update(status='failed', exit_status=1, error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.time()-start
        atomic_json(out/'run.json', report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report)+'\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
