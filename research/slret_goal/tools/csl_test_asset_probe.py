"""Fixed DEV input replay and CSL TEST metadata census; no TEST forward."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from extraction_resume import atomic_json
from inventory import ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    cli = parser.parse_args()
    out = ROOT / 'artifacts/slret_goal' / cli.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = dict(run_id=cli.run_id, status='running', pid=os.getpid(),
                  command=sys.argv, script_sha256=sha(__file__),
                  test_annotation_accessed=True, test_scores_accessed=False,
                  test_features_computed=False, optimizer_updates=0)
    atomic_json(out / 'run.json', report)
    try:
        assert shutil.disk_usage(out).free > 15 * 1024**3
        assert sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file()) + 16 * 1024**2 < 44 * 1024**3
        sys.path.insert(0, str(ROOT / 'shared'))
        from slr_common.features.i3d import (
            ExtractionRecipe, decode_video, infer_video_features, load_i3d,
            sliding_window_starts, _temporal_metadata,
        )
        from slr_common.csl_transfer import _load_transformer
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        device = torch.device('cuda:0')
        devpath = ROOT / 'artifacts/manifests/csl_dev.jsonl'
        devrows = [json.loads(line) for line in devpath.read_text().splitlines()]
        sample = devrows[0]
        rawroot = Path('/home/dongvk/datasets/CSL_Daily_Sentence_Crop')
        raw = rawroot / 'videos' / (sample['video_id'] + '.mp4')
        recipe = ExtractionRecipe()
        impl = ROOT / 'third_party/SLRT/CiCo/I3D_feature_extractor/models/i3d.py'
        report['source_hashes'] = {str(p): sha(p) for p in [
            devpath, impl, ROOT/'shared/slr_common/features/i3d.py',
            ROOT/'shared/slr_common/csl_transfer.py',
            ROOT/'research/slret_goal/CSL_TEST_ASSET_PROTOCOL.md']}
        frames, fps = decode_video(raw, recipe)
        starts = sliding_window_starts(len(frames), recipe.clip_frames, recipe.stride)
        temporal = _temporal_metadata(raw, sha(raw), len(frames), fps, starts, recipe)
        assert temporal == json.loads(Path(sample['temporal_metadata']).read_text())
        report['dev_feature_replays'] = []
        for stream in ['agnostic', 'aware']:
            reference = Path(sample[f'feature_{stream}'])
            meta = json.loads(reference.with_suffix('.pkl.meta.json').read_text())
            checkpoint = Path(meta['checkpoint'])
            assert sha(checkpoint) == meta['checkpoint_sha256']
            assert sha(reference) == meta['feature_sha256']
            assert meta['source_video_sha256'] == sha(raw)
            assert meta['recipe_sha256'] == recipe.digest
            assert meta['effective_batch_size'] == 128
            model = load_i3d(checkpoint, impl, device)
            torch.cuda.synchronize()
            tick = time.time()
            features, batch = infer_video_features(model, frames, starts, recipe, device, 128)
            torch.cuda.synchronize()
            elapsed = time.time() - tick
            with reference.open('rb') as f:
                expected = pickle.load(f)['feature']
            delta = float(np.max(np.abs(features - expected)))
            entry = dict(stream=stream, video_id=sample['video_id'], windows=len(starts),
                         seconds=elapsed, batch_size=batch, max_abs_delta=delta,
                         bit_exact=np.array_equal(features, expected),
                         reference=str(reference), reference_sha256=sha(reference),
                         checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
            report['dev_feature_replays'].append(entry)
            atomic_json(out/'run.json', report)
            assert batch == 128 and entry['bit_exact'], entry
            np.save(out/f'dev_{stream}.npy', features)
            del model, features, expected
            gc.collect()
            torch.cuda.empty_cache()
        del frames
        transpath = ROOT / 'artifacts/transfer/csl_daily_dev_en_opus_mt.json'
        trans = json.loads(transpath.read_text())
        modelroot = Path(trans['model']['root'])
        for name, expected_hash in trans['model']['file_sha256'].items():
            assert sha(modelroot/name) == expected_hash, name
        assert trans['generation'] == dict(batch_size=64, device='cuda:0',
            max_new_tokens=128, max_source_tokens=128, num_beams=4)
        translator = _load_transformer(modelroot, device, 64)
        rows = trans['translations'][:64]
        tick = time.time()
        translated = translator([r['source_text'] for r in rows])
        expected = [r['target_text'] for r in rows]
        report['translation_replay'] = dict(rows=len(rows), seconds=time.time()-tick,
            exact=translated == expected, mismatch_count=sum(a != b for a,b in zip(translated,expected)),
            source=str(transpath), source_sha256=sha(transpath), model=trans['model'])
        atomic_json(out/'run.json', report)
        assert translated == expected, report['translation_replay']
        del translator
        gc.collect()
        torch.cuda.empty_cache()
        testpath = rawroot/'test_data_with_num_frames.csv'
        with testpath.open() as f:
            test = list(csv.DictReader(f))
        ids = [r['name'] for r in test]
        groups = {}
        for row in test:
            cid = row['name'].split('_')[0]
            assert groups.setdefault(cid,row['text']) == row['text']
            assert (rawroot/'videos'/(row['name']+'.mp4')).is_file()
        assert len(ids) == len(set(ids)) == 1176
        overlap = {}
        for split in ['train','dev']:
            source = ROOT/f'artifacts/manifests/csl_{split}.jsonl'
            records = [json.loads(line) for line in source.read_text().splitlines()]
            videos = set(ids) & {r['video_id'] for r in records}
            overlap[split] = dict(video_ids=len(videos),
                caption_ids=len(set(groups)&{r['caption_id'] for r in records}))
            assert not videos
            report['source_hashes'][str(source)] = sha(source)
        windows = sum(max(1,int(r['frame_count'])-15) for r in test)
        rawbytes = windows*1024*4*2
        report['test_census'] = dict(videos=len(ids),caption_groups=len(groups),
            raw_present=len(ids),windows=windows,dual_stream_raw_bytes=rawbytes,
            projected_with_10pct_margin_bytes=int(rawbytes*1.1),overlap=overlap,
            annotation=str(testpath),annotation_sha256=sha(testpath))
        report.update(status='completed',exit_status=0,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      decision='DEV input parity verified; TEST metadata only. Full preparation needs bounded admission.')
    except Exception:
        report.update(status='failed', exit_status=1,error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds'] = time.time()-started
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report)+'\n')
        print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
