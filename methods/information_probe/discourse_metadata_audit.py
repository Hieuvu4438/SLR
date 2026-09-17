"""Read-only PH train/dev metadata census; inferred adjacency is NOT discourse truth.

No captions are used as model input, no raw video/features/scores are loaded, and
no test rows or files are opened. This is a pre-candidate boundary audit, not C43.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/proposal7/evidence/autonomous_search/POST_C42-DISCOURSE-METADATA.json'
SUFFIX = re.compile(r'^(.*)-([0-9]+)$')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inferred_index(identity):
    match = SUFFIX.fullmatch(identity)
    if match is None:
        return None
    return match[1], int(match[2])


def adjacency(identities):
    """Count filename suffix predecessor availability, not linguistic continuity."""
    indexed = {}
    for split, names in identities.items():
        for name in names:
            key = inferred_index(name)
            if key is not None:
                if key in indexed:
                    raise ValueError(f'Ambiguous numeric suffix key: {key}')
                indexed[key] = split
    result = {}
    for split, names in identities.items():
        counts = Counter()
        for name in names:
            key = inferred_index(name)
            if key is None:
                counts['unparsed'] += 1
                continue
            source, number = key
            previous = indexed.get((source, number - 1))
            counts['parsed'] += 1
            if previous is None:
                counts['predecessor_absent_from_inspected_train_dev'] += 1
            elif previous == split:
                counts['predecessor_same_split'] += 1
            else:
                counts['predecessor_other_inspected_split'] += 1
        result[split] = {k: counts[k] for k in (
            'parsed', 'unparsed', 'predecessor_same_split',
            'predecessor_other_inspected_split',
            'predecessor_absent_from_inspected_train_dev')}
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()
    inputs, records, split_reports = [], {}, {}

    def read_json(path, *, lines=False):
        data = path.read_bytes()
        inputs.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(data),
                       'bytes': len(data)})
        return [json.loads(x) for x in data.splitlines() if x] if lines else json.loads(data)

    for split, expected in [('train', 7096), ('dev', 519)]:
        manifest = read_json(ROOT / f'artifacts/manifests/ph_{split}.jsonl', lines=True)
        forensic = read_json(ROOT / f'artifacts/proposal7/forensics/ph_{split}.jsonl', lines=True)
        assert len(manifest) == len(forensic) == expected
        native = {x['id']: x for x in forensic}
        assert len(native) == expected
        assert set(native) == {x['video_id'] for x in manifest}
        schemas = {name: Counter() for name in ('manifest', 'forensic', 'temporal')}
        coordinates, verified_scopes = Counter(), Counter()
        agreement = Counter()
        source_paths, prefixes = set(), set()
        for row in manifest:
            identity = row['video_id']
            assert row['split'] == native[identity]['split'] == split
            assert row['pair_id'] == row['caption_id'] == identity
            inferred = inferred_index(identity)
            if inferred is not None:
                prefixes.add(inferred[0])
                agreement['forensic_source_matches_filename_prefix'] += int(native[identity]['source'] == inferred[0])
            temporal_path = Path(row['temporal_metadata']).resolve()
            expected_root = ROOT / f'artifacts/features_reextracted/ph_temporal_metadata/{split}'
            assert temporal_path.parent == expected_root and temporal_path.stem == identity
            temporal = read_json(temporal_path)
            for name, item in [('manifest', row), ('forensic', native[identity]), ('temporal', temporal)]:
                schemas[name][tuple(sorted(item))] += 1
            coordinates[temporal.get('coordinate_system', '<absent>')] += 1
            verified_scopes[temporal.get('verification_scope', '<absent>')] += 1
            source = Path(temporal['source_video'])
            source_paths.add(str(source))
            agreement['source_video_stem_matches_sentence_id'] += int(source.stem == identity)
            agreement['rf_start_begins_zero'] += int(bool(temporal['rf_start']) and temporal['rf_start'][0] == 0)
            agreement['rf_intervals_inside_clip'] += int(all(
                0 <= a < b <= temporal['decoded_frame_count']
                for a, b in zip(temporal['rf_start'], temporal['rf_end'])
            ) and len(temporal['rf_start']) == len(temporal['rf_end']))
        records[split] = [x['video_id'] for x in manifest]
        split_reports[split] = {
            'rows': expected, 'inferred_prefixes': len(prefixes),
            'unique_sentence_source_paths': len(source_paths),
            'schemas': {name: [{'keys': list(keys), 'rows': n} for keys, n in sorted(counter.items())]
                        for name, counter in schemas.items()},
            'coordinate_systems': dict(coordinates),
            'verification_scopes': dict(verified_scopes),
            'agreements': dict(agreement),
        }
    prefix_sets = {split: {inferred_index(x)[0] for x in ids if inferred_index(x) is not None}
                   for split, ids in records.items()}
    suffix_counts = adjacency(records)
    # Independent O(N^2) string equality implementation: no regex/index reuse.
    alternate = {}
    for split, ids in records.items():
        counts = Counter()
        for identity in ids:
            left, separator, right = identity.rpartition('-')
            assert separator and right.isascii() and right.isdigit()
            candidate = left + '-' + str(int(right) - 1)
            own = candidate in records[split]
            other = candidate in records['dev' if split == 'train' else 'train']
            assert not (own and other)
            counts['predecessor_same_split' if own else 'predecessor_other_inspected_split' if other
                   else 'predecessor_absent_from_inspected_train_dev'] += 1
        alternate[split] = dict(counts)
        assert all(suffix_counts[split][key] == value for key, value in counts.items())
    report = {
        'status': 'completed', 'code_sha256': digest(Path(__file__).read_bytes()),
        'scope': 'PH train/dev manifest, forensic metadata and every referenced temporal JSON only',
        'splits': split_reports, 'filename_suffix_adjacency_NOT_verified_context': suffix_counts,
        'independent_string_enumeration_agrees': True,
        'inferred_prefix_overlap_train_dev': len(prefix_sets['train'] & prefix_sets['dev']),
        'input_file_count': len(inputs), 'inputs': inputs,
        'prohibitions_preserved': ['no test access', 'no model or score evaluation', 'no feature/video loading',
                                  'no caption concatenation', 'no verified recording identity claim'],
        'method_go': False, 'wall_seconds': time.monotonic() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'inputs'}, indent=2))


if __name__ == '__main__':
    main()
