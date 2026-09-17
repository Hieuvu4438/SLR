"""Post-C44 permitted release metadata audit; no TEST files or model inputs."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
RELEASE = Path('/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3')
OUT = ROOT/'docs/proposal7/evidence/autonomous_search/POST_C44-ANNOTATION-COMPATIBILITY.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    inputs = []

    def read(path):
        raw = path.read_bytes()
        inputs.append({'path': str(path), 'bytes': len(raw),
                       'sha256': hashlib.sha256(raw).hexdigest()})
        return raw.decode('utf-8')

    readme = read(RELEASE/'README')
    read(RELEASE/'PHOENIX-2014-T/evaluation/README')
    exclusion = next(line for line in readme.splitlines()
                     if line.startswith('- No segments that have been annotated for mouthing'))
    assert 'present in any sets' in exclusion
    results, standard = {}, {}
    for split in ('train', 'dev'):
        manifest = [json.loads(line) for line in read(ROOT/f'artifacts/manifests/ph_{split}.jsonl').splitlines()]
        by_id = {row['video_id']: row for row in manifest}
        assert len(by_id) == len(manifest)
        assert all(row['split'] == split for row in manifest)
        variants = ['train', 'train-complex-annotation'] if split == 'train' else ['dev']
        for variant in variants:
            path = RELEASE/f'PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.{variant}.corpus.csv'
            source = read(path)
            reader = csv.DictReader(io.StringIO(source), delimiter='|')
            records = list(reader)
            # Independent parser must agree on every field, not just counts.
            lines = source.splitlines()
            fields = lines[0].split('|')
            assert all(len(line.split('|')) == len(fields) for line in lines[1:])
            alternate = [dict(zip(fields, line.split('|'))) for line in lines[1:]]
            assert alternate == records
            names = [r['name'] for r in records]
            assert len(set(names)) == len(names)
            if variant == split:
                assert set(names) == set(by_id)
                standard[split] = set(names)
            results[variant] = {
                'rows': len(records), 'schema': fields,
                'unique_ids': len(set(names)),
                'all_ids_match_corresponding_manifest': set(names) == set(by_id),
                'translation_exact_matches_manifest': sum(
                    r['name'] in by_id and r['translation'] == by_id[r['name']]['caption_original']
                    for r in records),
                'start_end_value_counts': [{'start': a, 'end': b, 'rows': n}
                                          for (a, b), n in sorted(Counter(
                                              (r['start'], r['end']) for r in records).items())],
                'video_path_is_sentence_id_plus_frame_glob': sum(
                    r['video'] == r['name']+'/1/*.png' for r in records),
                'csv_and_independent_field_parsers_exact': True,
            }
    assert not (standard['train'] & standard['dev'])
    report = {
        'status': 'completed', 'scope': 'two release READMEs, three permitted corpus CSVs, two manifests',
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol_sha256': hashlib.sha256((OUT.parent/'POST_C44_annotation_protocol.md').read_bytes()).hexdigest(),
        'inputs': inputs, 'corpora': results,
        'release_documentation_states_mouthing_annotation_sequences_excluded_from_all_sets': True,
        'this_statement_is_NOT_a_recomputed_footage_overlap_certificate': True,
        'train_dev_ids_disjoint': True, 'test_files_opened': 0,
        'archives_downloaded': 0, 'model_runs': 0, 'method_go': False,
        'wall_seconds': time.monotonic()-started,
    }
    with OUT.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
