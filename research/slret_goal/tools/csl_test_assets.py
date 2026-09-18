"""Prepare locked-recipe CSL TEST inputs in a new run; never score retrieval."""
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


def validate_feature(path, *, source_sha, recipe_sha, checkpoint_sha):
    sidecar=path.with_suffix('.pkl.meta.json')
    meta=json.loads(sidecar.read_text())
    if sha(path)!=meta['feature_sha256']:
        raise ValueError('Feature hash mismatch')
    if (meta['source_video_sha256']!=source_sha or meta['recipe_sha256']!=recipe_sha
            or meta['checkpoint_sha256']!=checkpoint_sha or meta['effective_batch_size']!=128):
        raise ValueError('Feature provenance or batch mismatch')
    with path.open('rb') as f:
        features=pickle.load(f)['feature']
    if (features.ndim!=2 or features.shape[1]!=1024 or features.dtype!=np.float32
            or not np.isfinite(features).all() or list(features.shape)!=meta['feature_shape']):
        raise ValueError('Invalid feature shape/dtype/values')
    return features.shape, {str(path):sha(path),str(sidecar):sha(sidecar)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    cli = parser.parse_args()
    root = ROOT/'artifacts/slret_goal'
    out = root/cli.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = dict(run_id=cli.run_id,status='running',pid=os.getpid(),command=sys.argv,
        script_sha256=sha(__file__),test_annotation_accessed=True,test_scores_accessed=False,
        optimizer_updates=0,stage='preflight')
    atomic_json(out/'run.json', report)
    try:
        probe_path = root/'csl-test-asset-probe-001/run.json'
        probe = json.loads(probe_path.read_text())
        assert probe['status'] == 'completed' and probe['translation_replay']['exact']
        assert all(r['bit_exact'] and r['batch_size']==128 for r in probe['dev_feature_replays'])
        # The admission addendum changed only the protocol document after probe.
        for path,expected in probe['source_hashes'].items():
            if not path.endswith('/CSL_TEST_ASSET_PROTOCOL.md'):
                assert sha(path)==expected, path
        assert shutil.disk_usage(out).free > 17*1024**3
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) + 2*1024**3 < 44*1024**3
        sys.path.insert(0,str(ROOT/'shared'))
        from slr_common.features.i3d import build_parser, extract, ExtractionRecipe
        from slr_common.csl_transfer import _load_transformer
        from slr_common.data.manifest import ManifestRecord, write_manifest, load_manifest
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        annotation = Path(probe['test_census']['annotation'])
        assert sha(annotation) == probe['test_census']['annotation_sha256']
        with annotation.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows)==1176 and len({r['name'] for r in rows})==1176
        groups = {}
        for row in rows:
            cid = row['name'].split('_')[0]
            assert groups.setdefault(cid,row['text'])==row['text']
        assert len(groups)==798
        rawroot = annotation.parent/'videos'
        listing = out/'test_videos.txt'
        listing.write_text(''.join(r['name']+'.mp4\n' for r in rows))
        protocol = ROOT/'research/slret_goal/CSL_TEST_ASSET_PROTOCOL.md'
        report['source_hashes'] = {**{p:sha(p) for p in probe['source_hashes']},
            str(probe_path):sha(probe_path),str(annotation):sha(annotation),
            str(listing):sha(listing),str(protocol):sha(protocol),str(Path(__file__).resolve()):sha(__file__)}
        report['probe_sha256']=sha(probe_path)
        report['test_census']=probe['test_census']
        report['stream_reports']={}
        recipe = ExtractionRecipe()
        for stream in ['agnostic','aware']:
            reference = next(x for x in probe['dev_feature_replays'] if x['stream']==stream)
            args = build_parser().parse_args([
                '--video-root',str(rawroot),'--checkpoint',reference['checkpoint'],
                '--expected-checkpoint-sha256',reference['checkpoint_sha256'],
                '--stream-name','domain_agnostic' if stream=='agnostic' else 'domain_aware_h2s_transfer',
                '--output-root',str(out/stream),'--temporal-metadata-root',str(out/'temporal'),
                '--splits','test','--split-video-list',f'test={listing}',
                '--batch-size','128','--device','cuda:0','--min-free-disk-gib','15',
                '--min-free-gpu-gib','8','--report-interval','25','--lock-wait-seconds','0'])
            report['stage']=f'extract_{stream}'
            atomic_json(out/'run.json',report)
            result=extract(args)
            report['stream_reports'][stream]=result
            assert result['status']=='complete' and result['completed']==1176 and not result['failed']
            assert result['effective_batch_size']==128 and result['recipe_sha256']==recipe.digest
            atomic_json(out/'run.json',report)
            gc.collect()
            torch.cuda.empty_cache()
        report['stage']='translate'
        atomic_json(out/'run.json',report)
        transpath=Path(probe['translation_replay']['source'])
        assert sha(transpath)==probe['translation_replay']['source_sha256']
        trans=json.loads(transpath.read_text())
        modelroot=Path(trans['model']['root'])
        for name,expected in trans['model']['file_sha256'].items():
            assert sha(modelroot/name)==expected, name
        translator=_load_transformer(modelroot,torch.device('cuda:0'),64)
        targets=translator(list(groups.values()))
        assert len(targets)==len(groups) and all(t.strip() for t in targets)
        translations={cid:t.strip() for cid,t in zip(groups,targets,strict=True)}
        translated=dict(schema_version=1,dataset='csl_daily',split='test',
            source_annotation=str(annotation),source_annotation_sha256=sha(annotation),
            model=trans['model'],generation=trans['generation'],
            test_annotation_accessed=True,reference_dev_artifact_sha256=sha(transpath),
            translations=[dict(caption_id=cid,source_text=source,target_text=translations[cid])
                          for cid,source in groups.items()])
        atomic_json(out/'translations.json',translated)
        del translator
        gc.collect()
        torch.cuda.empty_cache()
        report['stage']='validate_manifest'
        atomic_json(out/'run.json',report)
        records=[]
        asset_hashes={str(out/'translations.json'):sha(out/'translations.json')}
        for row in rows:
            vid=row['name']
            paths={s:out/s/'test'/(vid+'.pkl') for s in ['agnostic','aware']}
            temporal_path=out/'temporal/test'/(vid+'.json')
            temporal=json.loads(temporal_path.read_text())
            shapes=[]
            source_sha=sha(rawroot/(vid+'.mp4'))
            for stream,path in paths.items():
                expected=next(x for x in probe['dev_feature_replays'] if x['stream']==stream)
                shape,hashes=validate_feature(path,source_sha=source_sha,
                    recipe_sha=recipe.digest,checkpoint_sha=expected['checkpoint_sha256'])
                shapes.append(shape)
                asset_hashes.update(hashes)
            assert shapes[0]==shapes[1]
            # Temporal metadata is the unchanged extractor's native schema.
            assert len(temporal['rf_start'])==shapes[0][0]
            assert temporal['source_video_sha256']==source_sha and temporal['recipe_sha256']==recipe.digest
            assert temporal['verified'] is True
            asset_hashes[str(temporal_path)]=sha(temporal_path)
            cid=vid.split('_')[0]
            records.append(ManifestRecord(schema_version=1,dataset='csl_daily',split='test',
                pair_id=vid,video_id=vid,caption_id=cid,caption_original=row['text'],
                caption_model=translations[cid],caption_language='en',
                feature_agnostic=str(paths['agnostic']),feature_aware=str(paths['aware']),
                dense_length=shapes[0][0],feature_dim=1024,temporal_metadata=str(temporal_path)))
        manifest=out/'csl_test.jsonl'
        assert not manifest.exists()
        report['manifest']=dict(path=str(manifest),**write_manifest(records,manifest))
        assert len(load_manifest(manifest,expected_split='test'))==1176
        atomic_json(out/'asset_hashes.json',asset_hashes)
        report['asset_hashes_sha256']=sha(out/'asset_hashes.json')
        for path,expected in report['source_hashes'].items():
            assert sha(path)==expected, path
        assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<44*1024**3
        report.update(status='completed',exit_status=0,stage='complete',
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            decision='Inputs ready, no retrieval scored; separate fixed-model TEST lock required. DEV/TEST caption overlap795/798 limits novel-query inference.')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:
        report['wall_seconds']=time.time()-started
        atomic_json(out/'run.json',report)
        with (ROOT/'research/slret_goal/experiments.jsonl').open('a') as f:
            f.write(json.dumps(report)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='stream_reports'},indent=2))


if __name__=='__main__':
    main()
