"""CPU-only exact UPRet model/checkpoint bridge; no dataset or scoring."""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import torch

from inventory import ROOT, sha
from extraction_resume import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'CPU-only isolated check'
    out = ROOT / 'artifacts/slret_goal' / args.run_id
    out.mkdir(exist_ok=False)
    start = time.time()
    report = dict(run_id=args.run_id, status='running', command=sys.argv,
                  pid=os.getpid(), test_loaded=False, dataset_loaded=False,
                  gpu_inference=False, script_sha256=sha(__file__))
    try:
        torch.set_num_threads(4)
        sys.path.insert(0, str(ROOT))
        from methods.sssc.method1.config import load_config
        from methods.sssc.method1.model_factory import (
            _activate_upret_import, _load_clip_state, config_to_upret,
            _initialize_clip_weights, apply_upret_trainability, _seed_initialization)
        config_path = ROOT / 'methods/sssc/configs/method1/ph_seed42_base_initial.yaml'
        config = load_config(config_path)
        historical = ROOT / 'runs/method1/ph/base/seed42'
        manifest = json.loads((historical / 'run_manifest.json').read_text())
        assert config.digest == manifest['config_sha256'], 'Historical config changed'
        checkpoint = historical / 'best_dev.pt'
        expected = 'c72ecbc4ce35bec1f5fe53692a7d3f88b27e591c4692a93e82df26fdde00ff82'
        assert sha(checkpoint) == expected
        # Vendored layout has no nested .git. Verify actual source contents instead
        # of claiming that the parent SLR commit is the historical upstream commit.
        source_checks = {}
        for name, expected_source in manifest['artifact_hashes']['implementation_source']['files'].items():
            if not name.startswith('upret/') or name.endswith('modeling_clip4clip.py'):
                continue  # Different entrypoint, not imported by modules.modeling.
            path = ROOT/'third_party/UPRet'/name.removeprefix('upret/')
            content = path.read_bytes()
            normalized = content.replace(b'        # pdb.set_trace()', b'        pdb.set_trace()') if name.endswith('module_cross.py') else content
            assert hashlib.sha256(normalized).hexdigest() == expected_source, ('Source mismatch',name)
            source_checks[name] = sha(path)
        upret_root = ROOT/'third_party/UPRet'
        CLIP4Clip, CrossConfig = _activate_upret_import(upret_root)
        native_args = config_to_upret(config)
        _seed_initialization(config.seed)
        clip = _load_clip_state(config.model.clip_checkpoint_path)
        assert sha(config.model.clip_checkpoint_path) == manifest['artifact_hashes']['clip_initialization']
        cross = CrossConfig(str(upret_root/'modules/cross-base/cross_config.json'))
        model = CLIP4Clip(cross, clip.copy(), task_config=native_args).float()
        _initialize_clip_weights(model, clip)
        model.sample_num = config.model.distribution_samples
        model.eps = config.model.sample_ot_epsilon
        model.max_iter = config.model.sample_ot_max_iterations
        model.ot_weight = config.model.sample_ot_logit_weight
        apply_upret_trainability(model, config.model.freeze_layer_num)
        # Trusted local checkpoint identity verified above; envelope contains RNG objects.
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = payload['student_state_dict']
        model.load_state_dict(state, strict=True)
        actual = model.state_dict()
        assert set(actual) == set(state)
        assert all(torch.equal(actual[k], value) for k, value in state.items())
        report.update(status='completed', exit_status=0, tensor_count=len(state),
                      tensor_parity='exact', config_sha256=config.digest,
                      checkpoint_sha256=expected, checkpoint_step=payload['global_step'],
                      training_run_complete=payload['training_run_complete'],
                      provenance_mode='vendored_file_hashes_not_nested_git_revision',
                      verified_upret_source_sha256=source_checks,
                      allowed_source_difference='module_cross debugger disabled; no numerical change',
                      source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [
                          ROOT/'methods/sssc/method1/model_factory.py',
                          ROOT/'third_party/UPRet/modules/modeling.py',
                          ROOT/'third_party/UPRet/modules/module_cross.py']},
                      decision='Exact partial corrected checkpoint bridge only; no retrieval parity yet.')
    except Exception:
        report.update(status='failed', exit_status=1, error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.time()-start
        atomic_json(out/'run.json', report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as stream:
            stream.write(json.dumps(report)+'\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
