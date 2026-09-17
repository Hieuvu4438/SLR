"""AS-C45 exact gloss inventory, not linguistic equivalence or new positives."""
from collections import defaultdict
import csv
import itertools
import json
import os
import time
import traceback

import numpy as np

from .annotation_compatibility_audit import RELEASE
from .common import ROOT, dump, ranks, rows, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def exact_pairs(sequences):
    groups = defaultdict(list)
    for i, sequence in enumerate(sequences):
        if not sequence:
            raise ValueError('Empty gloss cannot establish a collision')
        groups[tuple(sequence)].append(i)
    return sorted(pair for indexes in groups.values()
                  for pair in itertools.combinations(indexes, 2)), groups


def main():
    output = OUT/'AS-C45-GLOSS_run.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    report = {'status': 'running', 'pid': os.getpid(), 'code_sha256': sha(__file__),
              'protocol_sha256': sha(OUT/'AS-C45_protocol.md'), 'inputs': {},
              'test_loaded': False, 'training_updates': 0, 'method_go': False}
    dump(output, report)
    try:
        inventories, dev_mask = {}, None
        for split in ('train', 'dev'):
            manifest_path = ROOT/f'artifacts/manifests/ph_{split}.jsonl'
            annotation_path = RELEASE/f'PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.{split}.corpus.csv'
            for p in (manifest_path, annotation_path):
                report['inputs'][str(p)] = sha(p)
            manifest = rows(split)
            with annotation_path.open() as f:
                native_rows = list(csv.DictReader(f, delimiter='|'))
            by_id = {r['name']: r for r in native_rows}
            assert len(by_id) == len(native_rows) == len(manifest)
            assert set(by_id) == {r['video_id'] for r in manifest}
            gloss = [tuple(by_id[r['video_id']]['orth'].split()) for r in manifest]
            assert all(by_id[r['video_id']]['translation'] == r['caption_original'] for r in manifest)
            pairs, groups = exact_pairs(gloss)
            # O(N^2) independent comparison; no grouping-key/index implementation reuse.
            alternate = [(i, j) for i in range(len(gloss)) for j in range(i+1, len(gloss))
                         if len(gloss[i]) == len(gloss[j]) and
                         all(a == b for a, b in zip(gloss[i], gloss[j]))]
            assert pairs == alternate
            different_native = [(i, j) for i, j in pairs
                                if manifest[i]['caption_original'] != manifest[j]['caption_original']]
            different_both = [(i, j) for i, j in different_native
                              if manifest[i]['caption_model'] != manifest[j]['caption_model']]
            inventories[split] = {
                'rows': len(manifest), 'unique_gloss_sequences': len(groups),
                'repeated_groups': sum(len(g) > 1 for g in groups.values()),
                'rows_in_repeated_groups': sum(len(g) for g in groups.values() if len(g) > 1),
                'exact_gloss_pairs': len(pairs),
                'different_native_translation_pairs': len(different_native),
                'different_native_and_model_translation_pairs': len(different_both),
                'independent_pair_enumeration_exact': True,
                'different_both_pair_ids': [[manifest[i]['video_id'], manifest[j]['video_id']]
                                            for i, j in different_both],
            }
            if split == 'dev':
                dev_mask = np.zeros((len(manifest), len(manifest)), dtype=bool)
                for i, j in different_both:
                    dev_mask[i, j] = dev_mask[j, i] = True
        scores, rank_records = [], []
        for seed in (42, 1337, 2026):
            path = ROOT/f'runs/ph_base_b512_s{seed}/evaluation/dev/scores_video_x_text.npy'
            report['inputs'][str(path)] = sha(path)
            score = np.load(path, allow_pickle=False)
            assert score.shape == dev_mask.shape and np.isfinite(score).all()
            scores.append(score)
            rank_records.append(ranks(score))
        directions = {}
        for direction, axis in [('T2V', 0), ('V2T', 1)]:
            persistent = np.stack([r[direction] > 0 for r in rank_records]).all(0)
            strict = np.stack([(s > (s.diagonal()[None, :] if axis == 0 else s.diagonal()[:, None]))
                               & dev_mask for s in scores])
            seed_query_hits = np.any(strict, axis=axis+1)
            all_seeds = seed_query_hits.all(0)
            same_competitor = strict.all(0).any(axis=axis)
            available = dev_mask.any(axis=axis)
            n = int(persistent.sum())
            hits = int((persistent & all_seeds).sum())
            directions[direction] = {
                'persistent_n': n,
                'all_query_candidate_availability': int(available.sum()),
                'persistent_candidate_availability': int((available & persistent).sum()),
                'persistent_strict_confuser_per_seed': [int((h & persistent).sum()) for h in seed_query_hits],
                'persistent_strict_confuser_all_seeds': hits,
                'persistent_same_strict_confuser_all_seeds': int((same_competitor & persistent).sum()),
                'persistent_fraction_all_seeds': hits/n if n else None,
                'registered_material_burden_pass': n > 0 and hits/n >= .1,
            }
        report.update(status='completed', inventories=inventories, directions=directions,
                      diagnostic_lead=all(r['registered_material_burden_pass'] for r in directions.values()),
                      wall_seconds=time.monotonic()-started)
        dump(output, report)
        print(json.dumps({**report, 'inventories': {k: {a:b for a,b in v.items() if a != 'different_both_pair_ids'}
                                                   for k,v in inventories.items()}}, indent=2))
    except Exception:
        report.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.monotonic()-started)
        dump(output, report)
        raise


if __name__ == '__main__':
    main()
