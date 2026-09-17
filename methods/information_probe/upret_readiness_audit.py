"""Read-only source/run-manifest census; never loads a model or dataset."""
import hashlib
import json
import time

from .common import ROOT, dump, sha


def main():
    started = time.monotonic()
    output = ROOT / 'docs/proposal7/evidence/autonomous_search/UPRET-READINESS_run.json'
    if output.exists():
        raise FileExistsError(output)
    run = ROOT / 'runs/method1/ph/base/seed42'
    manifest_path = run / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    expected = manifest['artifact_hashes']['implementation_source']['files']
    roots = {
        'method1': ROOT / 'methods/sssc/method1',
        'upret': ROOT / 'third_party/UPRet',
        'support': ROOT / 'methods/sssc',
    }
    checks = []
    for name, digest in sorted(expected.items()):
        prefix, relative = name.split('/', 1)
        if prefix == 'support' and relative.endswith('.patch'):
            relative = 'patches/' + relative
        path = roots[prefix] / relative
        actual = sha(path) if path.is_file() else None
        checks.append({'name': name, 'expected': digest, 'actual': actual,
                       'matches': actual == digest})
    records = []
    for path in sorted(run.glob('dev_step_*.json')):
        step = int(path.stem.removeprefix('dev_step_'))
        report = json.loads(path.read_text())
        t, v = report['T2V']['R1'], report['V2T']['R1']
        records.append({'step': step, 't2v_r1': t, 'v2t_r1': v,
                        'mean_r1': (t + v) / 2, 'sha256': sha(path)})
    records.sort(key=lambda item: item['step'])
    latest_path = run / 'latest_train_step.json'
    result = {
        'status': 'completed', 'code_sha256': sha(__file__),
        'run_manifest_sha256': sha(manifest_path),
        'source_checks': checks,
        'all_recorded_source_files_match': all(item['matches'] for item in checks),
        'source_check_scope': 'recorded files only; not a certificate of no added source files',
        'recorded_source_tree_sha256': manifest['artifact_hashes']['implementation_source']['content_sha256'],
        'training_complete_file_exists': (run / 'training_complete.json').is_file(),
        'declared_step_budget': manifest['effective_step_budget'],
        'latest_train_step': json.loads(latest_path.read_text()),
        'latest_train_step_sha256': sha(latest_path),
        'dev_records': records,
        'highest_saved_dev_mean': max(records, key=lambda item: item['mean_r1']),
        'last_saved_dev': records[-1],
        'checkpoint_loaded': False, 'dataset_loaded': False, 'test_loaded': False,
        'training_launched': False, 'method_go': False,
        'wall_seconds': time.monotonic() - started,
    }
    dump(output, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {'source_checks', 'dev_records'}}, indent=2))


if __name__ == '__main__':
    main()
