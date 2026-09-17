"""Independent scalar confuser validation for AS-C45; no model/TEST access."""
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import ROOT, dump, sha


def main():
    started = time.monotonic()
    folder = ROOT/'docs/proposal7/evidence/autonomous_search'
    path = folder/'AS-C45-GLOSS_run.json'
    output = folder/'AS-C45-VALIDATION_run.json'
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(path.read_text())
    assert report['status'] == 'completed'
    for file, digest in report['inputs'].items():
        assert sha(file) == digest
    manifest = [json.loads(line) for line in (ROOT/'artifacts/manifests/ph_dev.jsonl').read_text().splitlines()]
    annotation_path = next(Path(p) for p in report['inputs'] if p.endswith('T.dev.corpus.csv'))
    with annotation_path.open() as f:
        annotations = {r['name']: r for r in csv.DictReader(f, delimiter='|')}
    sequences = [annotations[r['video_id']]['orth'].split() for r in manifest]
    matrices = [np.load(ROOT/f'runs/ph_base_b512_s{s}/evaluation/dev/scores_video_x_text.npy', allow_pickle=False)
                for s in (42, 1337, 2026)]
    v_ranks = [torch.argsort(torch.argsort(torch.from_numpy(s), dim=1, descending=True), dim=1).diagonal().numpy()
               for s in matrices]
    checked = {}
    for direction in ('T2V', 'V2T'):
        persistent_n = available_n = all_n = same_n = 0
        per_seed = [0, 0, 0]
        for i, row in enumerate(manifest):
            is_persistent = all(any(s[j,i] > s[i,i] for j in range(len(manifest)))
                                for s in matrices) if direction == 'T2V' else all(r[i] > 0 for r in v_ranks)
            if not is_persistent:
                continue
            persistent_n += 1
            candidates = [j for j, other in enumerate(manifest) if j != i and sequences[j] == sequences[i]
                          and row['caption_original'] != other['caption_original']
                          and row['caption_model'] != other['caption_model']]
            available_n += bool(candidates)
            hit_sets = [{j for j in candidates if (s[j,i] if direction == 'T2V' else s[i,j]) > s[i,i]}
                        for s in matrices]
            for k, hits in enumerate(hit_sets):
                per_seed[k] += bool(hits)
            all_n += all(hit_sets)
            same_n += bool(set.intersection(*hit_sets))
        actual = {'persistent_n': persistent_n, 'persistent_candidate_availability': available_n,
                  'persistent_strict_confuser_per_seed': per_seed,
                  'persistent_strict_confuser_all_seeds': all_n,
                  'persistent_same_strict_confuser_all_seeds': same_n}
        assert all(report['directions'][direction][key] == val for key, val in actual.items())
        checked[direction] = actual
    assert report['diagnostic_lead'] == all(v['persistent_strict_confuser_all_seeds']/v['persistent_n'] >= .1
                                            for v in checked.values())
    result = {'status': 'completed', 'code_sha256': sha(__file__), 'source_run_sha256': sha(path),
              'input_hashes_exact': True, 'scalar_direction_counts_exact': checked,
              'scope': 'independent candidate/scalar loops; same historical torch tie convention',
              'method_go': False, 'test_loaded': False, 'wall_seconds': time.monotonic()-started}
    dump(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
