"""Independent length-bucket brute force for AS-C41 inventory, not score replay."""
from collections import defaultdict
from itertools import combinations
import json
import time

from .common import ROOT, dump, rows, sha
from .lexical_contrast_audit import OUT, words


def brute_force(sequences):
    by_length = defaultdict(list)
    for i, seq in enumerate(sequences):
        by_length[len(seq)].append(i)
    out = {}
    inspected = 0
    for indexes in by_length.values():
        for i, j in combinations(indexes, 2):
            inspected += 1
            different = []
            for p, (a, b) in enumerate(zip(sequences[i], sequences[j])):
                if a != b:
                    different.append(p)
                    if len(different) > 1:
                        break
            if len(different) == 1:
                out[i, j] = different[0]
    return out, inspected


def main():
    target = OUT/'AS-C41-INVENTORY-VALIDATION.json'
    if target.exists():
        raise FileExistsError(target)
    started = time.time()
    parent = OUT/'AS-C41-SINGLE-WORD_run.json'
    r = json.loads(parent.read_text())
    assert r['status'] == 'completed'
    report = {'status': 'validating', 'parent_run_sha256': sha(parent), 'code_sha256': sha(__file__),
              'scope': 'independent pair enumeration; same registered word normalization; no score/encoder rerun', 'splits': {}}
    for split in ('train', 'dev'):
        rec = rows(split)
        assert sha(ROOT/f'artifacts/manifests/ph_{split}.jsonl') == r['splits'][split]['manifest_sha256']
        en, en_n = brute_force([words(x['caption_model']) for x in rec])
        de, de_n = brute_force([words(x['caption_original']) for x in rec])
        old = r['splits'][split]
        assert len(en) == old['english_one_substitution_pair_n']
        assert len(de) == old['native_one_substitution_pair_n']
        pairs = {(x['i'], x['j']) for x in old['pair_details']}
        assert pairs == en.keys() & de.keys() and len(pairs) == old['bilingual_pair_n']
        assert all(en[x['i'], x['j']] == x['english_position'] and de[x['i'], x['j']] == x['native_position'] for x in old['pair_details'])
        report['splits'][split] = {'english_pair_n': len(en), 'native_pair_n': len(de), 'intersection_pair_n': len(pairs),
                                   'english_same_length_comparisons': en_n, 'native_same_length_comparisons': de_n,
                                   'intersection_ids_and_positions_exact': True}
    report.update(status='completed', wall_seconds=time.time()-started)
    dump(target, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
