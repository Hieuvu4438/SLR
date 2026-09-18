"""Read-only UPRet checkpoint/entrypoint provenance; no test or GPU inference."""
import json
import time
import torch

from inventory import ROOT, sha


def main():
    start = time.time()
    out = ROOT / 'artifacts/slret_goal/upret-asset-audit-001.json'
    if out.exists():
        raise FileExistsError(out)
    root = ROOT / 'runs/method1/ph/base/seed42'
    checkpoint_path = root / 'best_dev.pt'
    expected = 'c72ecbc4ce35bec1f5fe53692a7d3f88b27e591c4692a93e82df26fdde00ff82'
    actual = sha(checkpoint_path)
    assert actual == expected, 'Historical checkpoint identity changed'
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt['student_state_dict']
    manifest = json.loads((root / 'run_manifest.json').read_text())
    source_checks = []
    roots = {'method1': ROOT / 'methods/sssc/method1',
             'upret': ROOT / 'third_party/UPRet', 'support': ROOT / 'methods/sssc'}
    for name, digest in manifest['artifact_hashes']['implementation_source']['files'].items():
        prefix, relative = name.split('/', 1)
        if prefix == 'support' and relative.endswith('.patch'):
            relative = 'patches/' + relative
        path = roots[prefix] / relative
        source_checks.append(dict(name=name, recorded_sha256=digest,
                                 current_sha256=sha(path) if path.is_file() else None))
    report = dict(run_id='upret-asset-audit-001', status='completed', exit_status=0,
                  checkpoint=str(checkpoint_path), checkpoint_sha256=actual,
                  checkpoint_step=ckpt['global_step'], epoch=ckpt['epoch'],
                  training_run_complete=ckpt['training_run_complete'],
                  training_complete_marker=(root / 'training_complete.json').exists(),
                  declared_steps=manifest['effective_step_budget'],
                  latest_step=json.loads((root / 'latest_train_step.json').read_text())['global_step'],
                  tensor_count=len(state), distribution_keys=[k for k in state if 'dist_' in k],
                  state_container_key='student_state_dict',
                  native_init_expects='bare state_dict; does not unwrap student_state_dict',
                  native_entrypoint='third_party/UPRet/main_task_retrieval.py imports modules.modeling.CLIP4Clip',
                  source_checks=source_checks, script_sha256=sha(__file__),
                  manifest_sha256=sha(root / 'run_manifest.json'),
                  selection_split='historical_dev_selected_partial_checkpoint',
                  test_loaded=False, gpu_inference=False,
                  decision='Partial corrected local checkpoint available, not complete B_release. Do not pass its outer training envelope directly to native --init_model; require explicit mapping/strict tensor contract. No transport-reduction reopening.',
                  wall_seconds=time.time()-start)
    out.write_text(json.dumps(report, indent=2) + '\n')
    with (ROOT / 'research/slret_goal/experiments.jsonl').open('a') as f:
        f.write(json.dumps({k:v for k,v in report.items() if k not in ['source_checks','distribution_keys']} | {'artifact':str(out)}) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_checks','distribution_keys']}, indent=2))
    print('changed_recorded_sources', [r['name'] for r in source_checks if r['recorded_sha256'] != r['current_sha256']])


if __name__ == '__main__':
    main()
