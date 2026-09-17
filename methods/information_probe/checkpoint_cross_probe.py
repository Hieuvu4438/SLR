"""Cross best/last modalities to localize historical drift, not a method."""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from slr_common.data.cico_dataset import CiCoFeatureDataset
from slr_common.data.tokenize import CiCoCollator
from slr_common.upstream.cico_bridge import CiCoBridge, TextEncoding, VideoEncoding
from slr_common.upstream.factory import _load_cico_core_from_state, load_cico_tokenizer
from .common import ART, ROOT, dump, metrics, sha

OUT = ROOT/'docs/proposal7/evidence/autonomous_search'


@torch.inference_mode()
def encode(core, loader):
    bridge = CiCoBridge(core)
    videos, texts, ids = [], [], []
    for batch in loader:
        videos.append(bridge.encode_video(batch['h'].cuda(), batch['valid'].cuda()))
        texts.append(bridge.encode_text(*(x.cuda() for x in batch['clean_text'])))
        ids.extend(batch['pair_id'])
    return (VideoEncoding(*(torch.cat([getattr(v, k) for v in videos]) for k in ('mask','tokens','cls'))),
            TextEncoding(*(torch.cat([getattr(t, k) for t in texts]) for k in ('mask','tokens','cls'))), ids)


@torch.inference_mode()
def score(bridge, video, text):
    n = len(video.tokens)
    result = np.empty((n, n), np.float32)
    for i in range(0, n, 128):
        for j in range(0, n, 128):
            v = VideoEncoding(*(getattr(video,k)[i:i+128] for k in ('mask','tokens','cls')))
            t = TextEncoding(*(getattr(text,k)[j:j+128] for k in ('mask','tokens','cls')))
            a,b = bridge.score(v,t)
            result[i:i+128,j:j+128] = ((a+b)/2).cpu().numpy()
    return result


def main():
    report_path = OUT/'AS-C09-CROSS_run.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    torch.set_num_threads(8)
    start = time.time()
    result = {'experiment_id':'AS-C09-CROSS','status':'running','pid':os.getpid(),
              'code_sha256':sha(__file__),'trainable_parameters':0,'updates':0,
              'selector':None,'seeds':{},'method_go':False,
              'registered_conditions':['best_video_best_text','best_video_last_text','last_video_best_text','last_video_last_text']}
    dump(report_path,result)
    try:
        for seed in (42,1337,2026):
            run = ROOT/f'runs/ph_base_b512_s{seed}'
            config = yaml.safe_load((run/'resolved_config.yaml').read_text())
            cico_root = ROOT/config['upstream']['cico_root']
            sys.path.insert(0,str(cico_root))
            dataset = CiCoFeatureDataset(ROOT/'artifacts/manifests/ph_dev.jsonl',feature_len=64,alpha=.9,split='dev')
            loader = DataLoader(dataset,batch_size=128,num_workers=4,shuffle=False,
                                collate_fn=CiCoCollator(load_cico_tokenizer(config),32,augment=False))
            variants, hashes, scales = {}, {}, {}
            for name in ('best','last'):
                path = run/'checkpoints'/('best_dev.pt' if name=='best' else 'last.pt')
                hashes[name] = sha(path)
                saved = torch.load(path,weights_only=True,map_location='cpu',mmap=True)
                state = {k.removeprefix('core.'):v for k,v in saved['model'].items() if k.startswith('core.')}
                core = _load_cico_core_from_state(config,state,cico_root=cico_root,device='cuda')
                core.eval().requires_grad_(False)
                scales[name] = float(core.clip.logit_scale.exp())
                v,t,ids = encode(core,loader)
                variants[name] = (v,t)
                assert ids == [r.pair_id for r in dataset.records]
            # Only scalar logit_scale is used by the Filip scorer after encoding.
            core.clip.logit_scale.data.fill_(np.log(scales['best']))
            bridge = CiCoBridge(core)
            baseline = np.load(run/'evaluation/dev/scores_video_x_text.npy')
            entry = {'checkpoint_sha256':hashes,'logit_scale':scales,
                     'manifest_sha256':sha(ROOT/'artifacts/manifests/ph_dev.jsonl'),
                     'metrics':{},'drift':{}}
            for vi in ('best','last'):
                for ti in ('best','last'):
                    name = f'{vi}_video_{ti}_text'
                    s = score(bridge,variants[vi][0],variants[ti][1])
                    m = metrics(s,baseline)
                    if vi==ti=='best':
                        assert np.array_equal(s,baseline), 'best full-gallery replay must be exact'
                    prefix = f'AS-C09-s{seed}-{name}'
                    np.save(ART/f'{prefix}_scores.npy',s)
                    dump(OUT/f'{prefix}_metrics.json',m)
                    entry['metrics'][name] = {'T2V_R1':m['official_T2V']['R1'], 'V2T_R1':m['V2T']['R1'],
                                              'mean_R1':m['official_mean_R1'],
                                              'delta_mean_R1':m['official_mean_R1']-metrics(baseline)['official_mean_R1']}
            for k,name in enumerate(('video','text')):
                a,b = variants['best'][k],variants['last'][k]
                valid = a.mask== (0 if name=='video' else 1)
                assert torch.equal(a.mask,b.mask)
                distance = (1-F.cosine_similarity(a.tokens,b.tokens,dim=-1))[valid]
                entry['drift'][name] = {'mean_cosine_distance':float(distance.mean()),
                                        'median_cosine_distance':float(distance.median()),
                                        'p95_cosine_distance':float(torch.quantile(distance.float(),.95))}
            result['seeds'][str(seed)] = entry
            dump(report_path,result)
            print(json.dumps({'seed':seed,**entry}),flush=True)
            del variants,core
        result.update(status='completed',wall_seconds=time.time()-start,
                      limits=['Hybrid encoders can be coordinate-incompatible; effects do not uniquely identify harmful learning.',
                              'No calibration fitted and no checkpoint selected using these measurements.',
                              'Only endpoints, not intermediate representations, are available.',
                              'No new method claim; ordinary checkpoint averaging or frozen towers are controls, not novelty.'])
        dump(report_path,result)
    except Exception:
        result.update(status='failed',traceback=traceback.format_exc(),wall_seconds=time.time()-start)
        dump(report_path,result)
        raise


if __name__=='__main__':
    main()
