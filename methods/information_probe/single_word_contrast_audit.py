"""AS-C41 naturally occurring one-word changes corroborated in both captions."""
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
import time
import traceback

import numpy as np
import torch
import yaml

from slr_common.data.tokenize import encode_cico_text
from slr_common.upstream.factory import load_cico_tokenizer
from .common import ART, ROOT, dump, persistent, ranks, rows, sha
from .lexical_contrast_audit import OUT, margins_from_scores, pair_scores, words


def one_substitution_pairs(sequences):
    buckets = defaultdict(list)
    for i, seq in enumerate(sequences):
        for p in range(len(seq)):
            buckets[(p, seq[:p], seq[p+1:])].append(i)
    pairs = {}
    for (p, _, _), group in buckets.items():
        for i, j in combinations(group, 2):
            if sequences[i][p] != sequences[j][p]:
                assert (i, j) not in pairs
                pairs[i, j] = p
    return pairs


def bilingual_pairs(english, native):
    a, b = one_substitution_pairs(english), one_substitution_pairs(native)
    return a, b, sorted(a.keys() & b.keys())


def summarize_margins(measured):
    values = [x for e in measured for x in e['directional_margins'].values()]
    return {'pair_n': len(measured), 'directional_margin_n': len(values),
            'all_four_positive_pair_n': sum(e['all_four_strict'] for e in measured),
            'positive_margin_n': sum(x > 1e-4 for x in values),
            'near_zero_margin_n': sum(abs(x) <= 1e-4 for x in values),
            'negative_margin_n': sum(x < -1e-4 for x in values),
            'minimum_margin': min(values) if values else None,
            'maximum_margin': max(values) if values else None}


def persistent_coverage(scores, pairs):
    pair_set = {tuple(sorted(p)) for p in pairs}
    masks = persistent()
    out = {}
    wrong = scores.copy()
    np.fill_diagonal(wrong, -np.inf)
    for direction, axis in [('T2V', 0), ('V2T', 1)]:
        hard = wrong.argmax(axis=axis)
        covered = [i for i, j in enumerate(hard) if masks[direction][i] and tuple(sorted((i, int(j)))) in pair_set]
        negative = [i for i in covered if (scores[i, i]-(scores[hard[i], i] if axis == 0 else scores[i, hard[i]])) < -1e-4]
        out[direction] = {'persistent_n': int(masks[direction].sum()),
                          'eligible_fixed_confuser_query_n': len(covered), 'eligible_fixed_confuser_query_indices': covered,
                          'negative_margin_query_n': len(negative), 'negative_margin_query_indices': negative}
    return out


@torch.inference_mode()
def main():
    path = OUT/'AS-C41-SINGLE-WORD_run.json'
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(4)
    started = time.time()
    result = {'experiment_id': 'AS-C41-SINGLE-WORD', 'status': 'running', 'pid': os.getpid(),
              'code_sha256': sha(__file__), 'protocol_sha256': sha(OUT/'AS-C41_protocol.md'),
              'method_go': False, 'encoder_updates': 0, 'test_loaded': False, 'splits': {}}
    dump(path, result)
    try:
        parent_path = OUT/'AS-C32-ENSEMBLE_run.json'
        assert sha(parent_path) == 'bb16708f342f6b6cba64aa8c2949df4e4034375967b95e6ceaf5ff18767a3f11'
        parent = json.loads(parent_path.read_text())
        result['parent_run_sha256'] = sha(parent_path)
        config = yaml.safe_load((ROOT/'runs/ph_base_b512_s42/resolved_config.yaml').read_text())
        tok = load_cico_tokenizer(config)
        result['source_hashes'] = {str(p.relative_to(ROOT)): sha(p) for p in (
            ROOT/'methods/information_probe/lexical_contrast_audit.py',
            ROOT/'methods/information_probe/direction_probe.py', ROOT/'shared/slr_common/data/tokenize.py')}
        for split in ('train', 'dev'):
            if time.time()-started > 300:
                raise TimeoutError('AS-C41timeout300s')
            rec = rows(split)
            native_path = ROOT/f'artifacts/proposal7/forensics/ph_{split}.jsonl'
            metadata = {x['id']: x for x in map(json.loads, native_path.read_text().splitlines())}
            assert set(metadata) == {r['pair_id'] for r in rec}
            english, native = [words(r['caption_model']) for r in rec], [words(r['caption_original']) for r in rec]
            assert all(words(metadata[r['pair_id']]['text']) == native[i] and metadata[r['pair_id']]['split'] == split for i, r in enumerate(rec))
            a, b, both = bilingual_pairs(english, native)
            keys = [tuple(tuple(x.tolist()) for x in encode_cico_text(r['caption_model'], tok, 32)) for r in rec]
            eligible = [(i, j) for i, j in both if keys[i] != keys[j]]
            groups = defaultdict(list)
            details = []
            for i, j in both:
                ep, np_ = a[i, j], b[i, j]
                et = english[i][:ep]+('<SUB>',)+english[i][ep+1:]
                nt = native[i][:np_]+('<SUB>',)+native[i][np_+1:]
                groups[(et, nt)].append((i, j))
                details.append({'i': i, 'j': j, 'english_position': ep, 'native_position': np_,
                    'english_substitution': [english[i][ep], english[j][ep]],
                    'native_substitution': [native[i][np_], native[j][np_]],
                    'english_template': list(et), 'native_template': list(nt),
                    'different_deployed_inputs': keys[i] != keys[j],
                    'same_source': metadata[rec[i]['pair_id']]['source'] == metadata[rec[j]['pair_id']]['source'],
                    'same_signer': metadata[rec[i]['pair_id']]['signer'] == metadata[rec[j]['pair_id']]['signer']})
            cache_path = ART/f'frozen_{split}.pt'
            expected_cache = {'train': '9e0169b33b01f1120997818e550f3b9969effcc6e8af0e67164f3ec7d1d83062',
                              'dev': 'cb98214bf684a6f81c9a269d7c58f25c71620d3b955063beacf6cc8fce994cd1'}[split]
            assert sha(cache_path) == expected_cache
            cache = torch.load(cache_path, weights_only=True, map_location='cpu')
            assert cache['ids'] == [r['pair_id'] for r in rec]
            assert cache['manifest_sha256'] == sha(ROOT/f'artifacts/manifests/ph_{split}.jsonl')
            assert cache['checkpoint_sha256'] == parent['inputs']['42']['checkpoint_sha256']
            measured = pair_scores(cache, eligible)
            entry = {'row_n': len(rec), 'manifest_sha256': cache['manifest_sha256'], 'cache_sha256': expected_cache,
                     'checkpoint_sha256': cache['checkpoint_sha256'], 'native_forensics_sha256': sha(native_path),
                     'english_one_substitution_pair_n': len(a), 'native_one_substitution_pair_n': len(b),
                     'bilingual_pair_n': len(both), 'eligible_pair_n': len(eligible),
                     'bilingual_row_n': len({i for p in both for i in p}),
                     'eligible_row_n': len({i for p in eligible for i in p}),
                     'same_deployed_input_pair_n': len(both)-len(eligible), 'joint_template_n': len(groups),
                     'template_pair_counts': [{'english_template': list(k[0]), 'native_template': list(k[1]), 'pair_n': len(v)} for k, v in groups.items()],
                     'metadata_pair_counts': {'same_source': sum(e['same_source'] for e in details), 'same_signer': sum(e['same_signer'] for e in details)},
                     'pair_details': details,
                     'involved_rows': {str(i): {k: rec[i][k] for k in ('pair_id', 'video_id', 'caption_model', 'caption_original')} for i in sorted({i for p in both for i in p})}}
            if split == 'train':
                entry['R0_pair_summary'] = summarize_margins(measured)
                entry['R0_pairs'] = measured
            else:
                entry['models'] = {}
                matrices = {}
                for seed in (42, 1337, 2026):
                    p = ROOT/f'runs/ph_base_b512_s{seed}/evaluation/dev/scores_video_x_text.npy'
                    assert sha(p) == parent['inputs'][str(seed)]['score_sha256']
                    matrices[str(seed)] = np.load(p)
                p = ART/'AS-C32-uniform3_scores.npy'
                assert sha(p) == parent['ensemble_score_sha256']
                matrices['ensemble'] = np.load(p)
                exact_pairs = margins_from_scores(matrices['42'], eligible)
                errors = [abs(x-y) for e, f in zip(measured, exact_pairs) for x, y in zip(e['four_scores_aa_ab_ba_bb'], f['four_scores_aa_ab_ba_bb'])]
                max_error = max(errors) if errors else None
                assert max_error is None or max_error <= 5e-5
                entry['paired_kernel_check'] = {'pair_n': len(eligible), 'scalar_n': len(errors), 'max_abs_error': max_error, 'tolerance': 5e-5}
                for name, s in matrices.items():
                    pairs = margins_from_scores(s, eligible)
                    rr = ranks(s)
                    for e in pairs:
                        e['full_gallery_ranks'] = {d: [int(rr[d][e['i']]), int(rr[d][e['j']])] for d in rr}
                    entry['models'][name] = {'summary': summarize_margins(pairs), 'pairs': pairs,
                                             'persistent_fixed_confuser_coverage': persistent_coverage(s, eligible)}
            result['splits'][split] = entry
            dump(path, result)
            print(json.dumps({'split': split, 'english_pair_n': len(a), 'native_pair_n': len(b), 'bilingual_pair_n': len(both),
                              'eligible_pair_n': len(eligible), 'joint_template_n': len(groups),
                              'R0_pair_summary': summarize_margins(measured)}), flush=True)
            del cache
        result.update(status='completed', wall_seconds=time.time()-started,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated())
        dump(path, result)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc(), wall_seconds=time.time()-started)
        dump(path, result)
        raise


if __name__ == '__main__':
    main()
