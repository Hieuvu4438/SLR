"""Same-pretraining spatial-information acquisition, PH train/dev RGB only.

Eight baseline-preprocessed windows per video; retain final I3D 2x2 spatial maps
and global pooling controls. Verify pooled embeddings against the existing
agnostic feature stream at exactly the sampled window indices. A mismatch fails
closed, rather than being misinterpreted as new visual information.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from slr_common.data.cico_dataset import load_dense_feature
from slr_common.features.i3d import ExtractionRecipe, decode_video, infer_video_features, load_i3d
from .common import ART, ROOT, dump, rows, sha

OUT=ROOT/'docs/proposal7/evidence/autonomous_search'


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--splits',nargs='+',choices=['train','dev'],default=['dev','train'])
    p.add_argument('--limit',type=int,default=0,help='smoke only; incomplete caches cannot be used by training')
    args=p.parse_args()
    torch.set_num_threads(4)
    start=time.time()
    ckpt=ROOT/'artifacts/pretrained/bsl5k.pth.tar'
    impl=ROOT/'third_party/SLRT/CiCo/I3D_feature_extractor/models/i3d.py'
    expected='6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f'
    assert sha(ckpt)==expected
    expid='AS-C02-RAW-SMOKE-B32' if args.limit else 'AS-C02-RAW-CACHE-B32'
    report={'experiment_id':expid,'status':'running','pid':os.getpid(),'start_unix':start,
            'command':sys.argv,'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'checkpoint_sha256':expected,'implementation_sha256':sha(impl),'code_sha256':sha(__file__),
            'pretraining':'EXACT baseline agnostic bsl5k I3D weights; no new asset/pretraining',
            'gpu':torch.cuda.get_device_name(),'precision':'float32 inference; float32 saved spatial maps',
            'retained_windows_per_video':8,'spatial_grid':[2,2],
            'sampling':'linspace over original dense-window indices; execute complete historical batches before selecting',
            'extraction_batch':32,'processing_control':'exact same full-window extraction for spatial and pooled arms',
            'complete_cache':not bool(args.limit),'splits':{},'failed':[]}
    dump(OUT/f'{expid}_run.json',report)
    device=torch.device('cuda')
    model=load_i3d(ckpt,impl,device)
    recipe=ExtractionRecipe()
    activation=[]
    def capture(module, inp, output):
        activation.append(F.adaptive_avg_pool3d(output,(1,2,2)).squeeze(2).flatten(2).transpose(1,2).cpu())
    hook=model._modules['Mixed_5c'].register_forward_hook(capture)
    report['frozen_parameters']=sum(p.numel() for p in model.parameters())
    report['trainable_parameters']=0
    try:
        for split in args.splits:
            records=rows(split)
            selected=records[:args.limit] if args.limit else records
            ids,grids,pools,provenance=[],[],[],[]
            for i,r in enumerate(selected):
                meta_path=Path(r['temporal_metadata'])
                meta=json.loads(meta_path.read_text())
                source=Path(meta['source_video'])
                # Exact path metadata and content digest bind each raw video.
                assert split in source.parts and source.stem==r['video_id']
                assert sha(source)==meta['source_video_sha256']
                assert meta['recipe_sha256']==recipe.digest
                indexes=np.linspace(0,r['dense_length']-1,8,dtype=int)
                # CuDNN/TF32 kernels depend on batch shape. The diagnostic
                # AS-C02-RAW-PARITY established exact equality at historical
                # batch32 and small drift for subset batches. Preserve the
                # strict gate and full historical batch composition instead
                # of relaxing tolerance after the failed smoke.
                starts=meta['rf_start']
                frames,fps=decode_video(source,recipe)
                assert len(frames)==meta['decoded_frame_count'] and abs(fps-meta['fps'])<1e-4
                activation.clear()
                features,_=infer_video_features(model,frames,starts,recipe,device,batch_size=32)
                features=features[indexes]
                spatial=torch.cat(activation)[indexes]
                assert spatial.shape==(8,4,1024) and torch.isfinite(spatial).all()
                old=load_dense_feature(r['feature_agnostic'])[indexes].numpy()
                absdiff=float(np.max(np.abs(features-old)))
                rel=float(np.linalg.norm(features-old)/max(np.linalg.norm(old),1e-12))
                if rel>1e-4 or absdiff>.01:
                    raise RuntimeError(f'Raw/old pooled parity failed {r["pair_id"]}: rel={rel}, abs={absdiff}')
                ids.append(r['pair_id']);grids.append(spatial);pools.append(torch.from_numpy(features))
                provenance.append({'id':r['pair_id'],'source_sha256':meta['source_video_sha256'],
                                   'metadata_sha256':sha(meta_path),'feature_sha256':sha(r['feature_agnostic']),
                                   'window_indexes':indexes.tolist(),'starts':[starts[int(j)] for j in indexes],
                                   'pooled_max_abs_delta':absdiff,'pooled_relative_delta':rel})
                if i%25==0 or i+1==len(selected):
                    entry={'done':i+1,'total':len(selected),'wall_seconds':time.time()-start,
                           'last_pooled_relative_delta':rel}
                    report['splits'][split]=entry
                    dump(OUT/f'{expid}_run.json',report)
                    print(json.dumps({'event':'raw_cache','split':split,**entry}),flush=True)
                del frames
            data={'ids':ids,'spatial':torch.stack(grids),'pooled':torch.stack(pools),
                  'manifest_sha256':sha(ROOT/f'artifacts/manifests/ph_{split}.jsonl'),
                  'checkpoint_sha256':expected,'recipe_sha256':recipe.digest,
                  'complete':len(ids)==len(records),'provenance':provenance}
            output=ART/f'raw_{split}{"_smoke" if args.limit else ""}.pt'
            if output.exists():
                raise FileExistsError(f'Refusing to overwrite {output}')
            torch.save(data,output.with_suffix('.pt.tmp'))
            output.with_suffix('.pt.tmp').replace(output)
            report['splits'][split].update(cache_sha256=sha(output),cache_bytes=output.stat().st_size,
                                           manifest_sha256=data['manifest_sha256'])
            dump(OUT/f'{expid}_run.json',report)
        report['status']='completed'
    except Exception:
        report['status']='failed'
        report['failed'].append(traceback.format_exc())
        raise
    finally:
        hook.remove()
        report['wall_seconds']=time.time()-start
        report['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
        dump(OUT/f'{expid}_run.json',report)


if __name__=='__main__':
    main()
