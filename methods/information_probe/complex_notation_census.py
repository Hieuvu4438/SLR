"""Literal syntax census of existing PH complex TRAIN annotations only."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
RELEASE = Path('/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3')
DIRECTORY = RELEASE / 'PHOENIX-2014-T/annotations/manual'


def main():
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw.decode('utf-8')

    records = {}
    for name in ['train', 'train-complex-annotation']:
        rows = list(csv.DictReader(io.StringIO(read(DIRECTORY / f'PHOENIX-2014-T.{name}.corpus.csv')), delimiter='|'))
        records[name] = {row['name']: row for row in rows}
        assert len(records[name]) == len(rows) == 7096
    manifest = [json.loads(line) for line in read(ROOT / 'artifacts/manifests/ph_train.jsonl').splitlines()]
    assert all(row['split'] == 'train' for row in manifest)
    assert set(records['train']) == set(records['train-complex-annotation']) == {row['video_id'] for row in manifest}
    report = {}
    for name, rows in records.items():
        sequences = [row['orth'].split() for row in rows.values()]
        predicates = {}
        for prefix in ['lh-', 'bh-', 'neg-', 'negalp-', 'negalpha-', 'poss-', 'loc-', 'cl-']:
            predicates['prefix:' + prefix] = lambda t, p=prefix: t.startswith(p)
        for field in ['mb:', 'mk:', 'lh:', 'name:', 'time:', 'loc:', 'obj:', 'in:']:
            predicates['literal_field:' + field] = lambda t, f=field: f in t
        predicates.update({
            'contains_hash': lambda t: '#' in t,
            'contains_parenthesis': lambda t: '(' in t or ')' in t,
            'suffix_MINUS_PLUSPLUS': lambda t: t.endswith('-PLUSPLUS'),
            'contains_literal_plus': lambda t: '+' in t,
        })
        counts = {}
        for label, predicate in predicates.items():
            per_row = [sum(predicate(token) for token in tokens) for tokens in sequences]
            counts[label] = {'tokens': sum(per_row), 'rows': sum(n > 0 for n in per_row)}
        specials = Counter(token for tokens in sequences for token in tokens if re.fullmatch(r'__.*__', token))
        report[name] = {
            'rows': len(rows), 'tokens': sum(map(len, sequences)),
            'unique_tokens': len({token for tokens in sequences for token in tokens}),
            'literal_counts': counts,
            'special_tokens': {token: {'tokens': count, 'rows': sum(token in tokens for tokens in sequences)} for token, count in sorted(specials.items())},
            'all_start_end_minus_one': all(row['start'] == row['end'] == '-1' for row in rows.values()),
        }
    read(RELEASE / 'README')
    read(RELEASE / 'PHOENIX-2014-T/evaluation/sign-recognition/evaluatePHOENIX-2014-T-signrecognition.sh')
    report['input_sha256'] = hashes
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['scope'] = 'literal TRAIN annotation syntax, not model efficacy or newly inferred labels'
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
