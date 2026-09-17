"""Read only the allowlisted existing TRAIN audit and verify exact ID joins."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PIN = '012b11c22b1c64b79325ec6b004b81693db86187'
URL = f'https://raw.githubusercontent.com/DFKI-SignLanguage/sacre-bird-phoenix/{PIN}/train_annotations_sacrebirdphoenix.csv'


def run():
    request = urllib.request.Request(URL, headers={'User-Agent': 'SLRet-train-resource-audit'})
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.url == URL, 'Unregistered redirect'
        raw = response.read()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')), delimiter='|')
    rows = list(reader)
    fields = reader.fieldnames
    assert fields is not None and 'name' in fields
    assert all(None not in r for r in rows), 'Malformed extra columns'
    manifests = {}
    ids = {}
    for split in ('train', 'dev'):
        path = ROOT / f'artifacts/manifests/ph_{split}.jsonl'
        content = path.read_bytes()
        records = [json.loads(line) for line in content.splitlines()]
        assert all(r['split'] == split for r in records)
        ids[split] = {r['video_id'] for r in records}
        assert len(ids[split]) == len(records)
        manifests[split] = {'rows': len(records), 'sha256': hashlib.sha256(content).hexdigest()}
    assert not ids['train'] & ids['dev']
    names = [r['name'] for r in rows]
    counts = Counter(names)
    categories = [f for f in fields if f not in ('name', 'comment')]
    return {
        'status': 'completed', 'repository_commit': PIN, 'source_url': URL,
        'source_sha256': hashlib.sha256(raw).hexdigest(), 'source_bytes': len(raw),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'schema': fields, 'rows': len(rows), 'unique_ids': len(counts),
        'duplicate_ids': {k: n for k, n in counts.items() if n > 1},
        'train_matched_rows': sum(n in ids['train'] for n in names),
        'dev_matched_rows': sum(n in ids['dev'] for n in names),
        'non_train_ids': sorted(set(names) - ids['train']),
        'category_raw_value_counts': {f: dict(Counter(r[f] for r in rows)) for f in categories},
        'missing_value_counts': {f: sum(r[f] in ('', None) for r in rows) for f in fields},
        'manifests': manifests, 'test_files_read': 0, 'model_score_access': False,
        'comments_semantically_inspected': False, 'changed_labels_or_positives': False,
        'method_go': False,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
