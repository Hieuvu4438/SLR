"""Investigate the failed raw smoke without relaxing its parity gate."""
import json
from pathlib import Path

import numpy as np
import torch

from slr_common.data.cico_dataset import load_dense_feature
from slr_common.features.i3d import ExtractionRecipe,decode_video,infer_video_features,load_i3d
from .common import ROOT,dump,rows,sha


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    r=rows('dev')[0]
    meta=json.loads(Path(r['temporal_metadata']).read_text())
    frames,_=decode_video(Path(meta['source_video']),ExtractionRecipe())
    model=load_i3d(ROOT/'artifacts/pretrained/bsl5k.pth.tar',
                   ROOT/'third_party/SLRT/CiCo/I3D_feature_extractor/models/i3d.py',torch.device('cuda'))
    old=load_dense_feature(r['feature_agnostic']).numpy()
    indexes=np.linspace(0,len(old)-1,8,dtype=int)
    result={'id':r['pair_id'],'experiment_id':'AS-C02-RAW-PARITY',
            'code_sha256':sha(__file__),'variants':{},'default_tf32':torch.backends.cudnn.allow_tf32}
    for name,starts,batch,select in [
        ('subset_batch8',[meta['rf_start'][i] for i in indexes],8,None),
        ('full_batch32',meta['rf_start'],32,indexes),
        ('full_batch8',meta['rf_start'],8,indexes),
        ('subset_batch8_tf32off',[meta['rf_start'][i] for i in indexes],8,None),
    ]:
        if name.endswith('tf32off'):
            torch.backends.cudnn.allow_tf32=False
        f,_=infer_video_features(model,frames,starts,ExtractionRecipe(),torch.device('cuda'),batch)
        if select is not None:
            f=f[select]
        reference=old[indexes]
        result['variants'][name]={'max_abs':float(np.abs(f-reference).max()),
                                 'relative_norm':float(np.linalg.norm(f-reference)/np.linalg.norm(reference))}
    dump(ROOT/'docs/proposal7/evidence/autonomous_search/AS-C02-RAW-PARITY.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
