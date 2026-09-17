"""Read-only TRAIN temporal-input census; never load videos or model assets."""
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest_path = ROOT / 'artifacts/manifests/ph_train.jsonl'
    manifest_bytes = manifest_path.read_bytes()
    manifest = [json.loads(line) for line in manifest_bytes.splitlines() if line]
    identities = [row['video_id'] for row in manifest]
    assert len(identities) == len(set(identities)) == 7096
    directory = ROOT / 'artifacts/features_reextracted/ph_temporal_metadata/train'
    assert {path.stem for path in directory.glob('*.json')} == set(identities)
    counts = Counter()
    frames, recipes, file_hashes = [], Counter(), []
    for row in sorted(manifest, key=lambda item: item['video_id']):
        assert row['split'] == 'train'
        path = Path(row['temporal_metadata']).resolve()
        assert path.parent == directory and path.stem == row['video_id']
        raw = path.read_bytes()
        file_hashes.append([path.name, sha(raw)])
        item = json.loads(raw)
        n = item['decoded_frame_count']
        assert isinstance(n, int) and n > 0
        assert item['coordinate_system'] == 'input_frame'
        assert item['interval_convention'] == 'half_open'
        assert item['verification_scope'] == 'generated_by_deterministic_elsc_i3d_extractor'
        recipe = item['recipe']
        recipes[(recipe['clip_frames'], recipe['stride'])] += 1
        assert recipe['clip_frames'] == 16 and recipe['stride'] == 1
        expected = list(range(max(n - 15, 1)))
        counts['start_sequence_mismatch'] += item['rf_start'] != expected
        counts['end_sequence_mismatch'] += item['rf_end'] != [min(s + 16, n) for s in expected]
        counts['below_16_frames'] += n < 16
        counts['equal_16_frames'] += n == 16
        counts['above_16_frames'] += n > 16
        counts['windows'] += len(item['rf_start'])
        counts['repeat_last_frame_draws_from_short_input'] += max(16 - n, 0)
        frames.append(n)
    sources = [
        'shared/slr_common/features/i3d.py',
        'third_party/SLRT/CiCo/I3D_feature_extractor/datasets/videodataset.py',
        'third_party/SLRT/CiCo/I3D_feature_extractor/datasets/phoenix2014.py',
        'methods/information_probe/input_padding_exposure.py',
    ]
    result = {
        'scope': 'TRAIN metadata only; no decode, features, model or scoring',
        'videos': len(frames), 'min_frames': min(frames), 'max_frames': max(frames),
        'counts': dict(counts),
        'recipes': [{'clip_frames': c, 'stride': s, 'videos': v} for (c, s), v in sorted(recipes.items())],
        'manifest_sha256': sha(manifest_bytes),
        'metadata_digest_definition': 'SHA256 of compact JSON sorted [basename, SHA256(bytes)] pairs',
        'metadata_digest': sha(json.dumps(file_hashes, separators=(',', ':')).encode()),
        'source_sha256': {p: sha((ROOT / p).read_bytes()) for p in sources},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
