"""Natural same-lexicon inventory; NOT validated sign-language minimal pairs."""
from collections import defaultdict
import itertools
import json
import os
import re
import time
import traceback

import numpy as np
import torch
import yaml

from slr_common.data.tokenize import encode_cico_text
from slr_common.upstream.factory import load_cico_tokenizer
from .common import ART, ROOT, dump, ranks, rows, sha
from .direction_probe import paired

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


def words(text):
    return tuple(re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.lower()))


def groups_with_order_contrasts(sequences):
    groups = defaultdict(list)
    for i, seq in enumerate(sequences):
        groups[tuple(sorted(seq))].append(i)
    return [ids for ids in groups.values() if len({sequences[i] for i in ids}) > 1]


def inventory(records, tokenizer):
    ordered_words = [words(r['caption_model']) for r in records]
    encoded = [encode_cico_text(r['caption_model'], tokenizer, 32) for r in records]
    ordered_bpe = [tuple(x[0][1:int(x[2].sum())-1].tolist()) for x in encoded]
    result, candidates = {}, set()
    for name, sequences in [('word_multiset', ordered_words), ('deployed_bpe_multiset', ordered_bpe)]:
        groups = groups_with_order_contrasts(sequences)
        pairs = []
        for ids in groups:
            for i, j in itertools.combinations(ids, 2):
                if sequences[i] != sequences[j]:
                    distinct = ordered_bpe[i] != ordered_bpe[j]
                    pairs.append({'i': i, 'j': j, 'different_deployed_input': distinct})
                    if distinct:
                        candidates.add((i, j))
        result[name] = {'group_n': len(groups), 'row_n': sum(map(len, groups)),
                        'pair_n': len(pairs), 'different_deployed_pair_n': sum(x['different_deployed_input'] for x in pairs),
                        'groups': [[{'index': i, 'pair_id': records[i]['pair_id'],
                                     'caption_model': records[i]['caption_model'],
                                     'caption_original': records[i]['caption_original'],
                                     'source_prefix': records[i]['video_id'].rsplit('-', 1)[0]} for i in ids] for ids in groups],
                        'pairs': pairs}
    return result, sorted(candidates)


@torch.inference_mode()
def pair_scores(cache, candidates):
    if not candidates:
        return []
    keys = sorted({(i, j) for a, b in candidates for i, j in ((a, a), (a, b), (b, a), (b, b))})
    scores = {}
    for start in range(0, len(keys), 128):
        chunk = keys[start:start+128]
        vi, ti = torch.tensor([x[0] for x in chunk]), torch.tensor([x[1] for x in chunk])
        scored = paired(cache['video_tokens'][vi].cuda(), cache['text_tokens'][ti].cuda(),
                        cache['video_mask'][vi].cuda(), cache['text_mask'][ti].cuda(), cache['logit_scale']).mean(-1)
        scores.update(zip(chunk, scored.cpu().double().tolist()))
    return margins_from_scores(scores, candidates)


def margins_from_scores(scores, candidates):
    results = []
    for a, b in candidates:
        aa, ab, ba, bb = (scores[i, j] for i, j in ((a, a), (a, b), (b, a), (b, b)))
        margins = {'V2T_a': aa-ab, 'V2T_b': bb-ba, 'T2V_a': aa-ba, 'T2V_b': bb-ab}
        results.append({'i': a, 'j': b, 'four_scores_aa_ab_ba_bb': [aa, ab, ba, bb],
                        'directional_margins': margins, 'all_four_strict': all(x > 1e-4 for x in margins.values())})
    return results


def main():
    path = OUT/'AS-C16-LEXICAL_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(4)
    started = time.time()
    report = {'experiment_id': 'AS-C16-LEXICAL', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C16_protocol.md'),
              'tokenize_sha256': sha(ROOT/'shared/slr_common/data/tokenize.py'),
              'paired_scorer_sha256': sha(ROOT/'methods/information_probe/direction_probe.py'),
              'method_go': False, 'fit_updates': 0, 'selector': None, 'splits': {}}
    dump(path, report)
    try:
        config = yaml.safe_load((ROOT/'runs/ph_base_b512_s42/resolved_config.yaml').read_text())
        tok = load_cico_tokenizer(config)
        for split in ('train', 'dev'):
            records = rows(split)
            collected, candidates = inventory(records, tok)
            cache = torch.load(ART/f'frozen_{split}.pt', weights_only=True)
            assert cache['ids'] == [r['pair_id'] for r in records]
            assert cache['manifest_sha256'] == sha(ROOT/f'artifacts/manifests/ph_{split}.jsonl')
            if split == 'train':
                measured = pair_scores(cache, candidates)
            else:
                matrix = np.load(ART/'baseline_dev_scores.npy')
                measured = margins_from_scores(matrix, candidates)
                rr = ranks(matrix)
                for entry in measured:
                    entry['original_full_gallery_ranks'] = {d: [int(rr[d][entry[k]]) for k in ('i', 'j')] for d in rr}
            entry = {'n': len(records), 'manifest_sha256': cache['manifest_sha256'],
                     'cache_sha256': sha(ART/f'frozen_{split}.pt'), 'checkpoint_sha256': cache['checkpoint_sha256'],
                     'inventory': collected, 'union_distinct_deployed_pairs': len(candidates),
                     'pairs_all_four_strict': sum(x['all_four_strict'] for x in measured), 'measured_pairs': measured}
            report['splits'][split] = entry
            dump(path, report)
            print(json.dumps({'split': split, 'inventory': {n: {k: v for k, v in x.items() if k not in ('groups', 'pairs')}
                                                            for n, x in collected.items()},
                              'union_pairs': len(candidates), 'all_four_strict': entry['pairs_all_four_strict']}), flush=True)
            del cache
        report.update(status='completed', wall_seconds=time.time()-started,
                      limits=['Same lexical multiset is not proof of a relational minimal contrast.',
                              'Text interpretation is not signed-video validity or new relevance labeling.',
                              'Pairwise accuracy is not full-gallery R1; no method trained or evaluated.',
                              'Negative availability screen is not absence of compositional information.'])
        dump(path, report)
    except Exception:
        report.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, report)
        raise


if __name__ == '__main__':
    main()
