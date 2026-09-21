"""C16 asset pilot: frozen public SignRep on exact native SEDS RGB windows.

Only TRAIN512 + full DEV519. No TEST, retrieval training, or model selection.
Per-video atomic cache supports an explicitly requested, contract-checked resume.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from methods.seds_adaptation.geometry_cache import selected_metadata, clip_centers, SELECTION_SALT


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4*1024**2), b''):h.update(part)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def make_plan(smoke=False):
    items = []
    for split, count in (('train',512),('dev',519)):
        root = ROOT/f'artifacts/slret_goal/seds-adapted-{split}-001/metadata'
        paths = selected_metadata(root,count)
        if smoke:paths = paths[:1]
        for path in paths:
            meta = json.loads(path.read_text())
            _, starts = clip_centers(meta)
            raw = Path(meta['raw_path'])
            if meta['id'] != path.stem or raw.parent.name != split or not raw.is_file():
                raise ValueError('Video identity/split mismatch')
            st = raw.stat()
            items.append(dict(id=path.stem,split=split,raw_path=str(raw),
                raw_bytes=st.st_size,raw_mtime_ns=st.st_mtime_ns,
                metadata_sha256=sha(path),decoded_frames=meta['decoded_frames'],
                clip_starts=starts.tolist(),windows=meta['original_frame_indices_per_window']))
    return dict(items=items,selection_salt=SELECTION_SALT,test_used=False,
                total_clips=sum(len(x['windows']) for x in items))


def load_model(source, checkpoint):
    import torch
    sys.path.insert(0,str(source))
    # Reuse upstream example's literal model parameters without executing its
    # example main, which contains placeholder paths and imports albumentations.
    from models.final_models.FINAL_hiera_latent_model_head_v25_active import Model
    dims = dict(full_body_angles=44,full_right_angles=82,full_left_angles=82,
                body_pose=183,right_pose=63,left_pose=63,body_dist=792,
                right_dist=165,left_dist=165)
    model = Model(ckpt_dir=None,backbone_params=dict(drop_path_rate=.1,num_cls_tokens=0),
        out_dim=768,mask_ratio=0.,init_weights=False,
        cls_head_name='models.head_models.dict_head_v6_FINAL_thoughtful',
        cls_head_params=dict(latents_params=dict(hidden_dim=512,out_dim=384),
            total_frames=16,projection_params={k:dict(hidden_dim=None,output_dim=v) for k,v in dims.items()}))
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
    result = model.load_state_dict(payload['model'],strict=True)
    if result.missing_keys or result.unexpected_keys:raise ValueError('Checkpoint mismatch')
    model.eval().requires_grad_(False)
    return model.cuda()


def decode(item):
    import cv2
    import numpy as np
    wanted = sorted({i for w in item['windows'] for i in w})
    frames = {}; needed = set(wanted)
    cap = cv2.VideoCapture(item['raw_path'])
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:break
            if count in needed:
                # Same deterministic evaluation resize/RGB/ImageNet normalization
                # as upstream A.Resize(224,224)+A.Normalize; no crops/augmentation.
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frames[count] = cv2.resize(frame,(224,224),interpolation=cv2.INTER_LINEAR)
            count += 1
    finally:cap.release()
    if count != item['decoded_frames'] or set(frames) != needed:
        raise ValueError('Decoded video/frame identity differs from inherited metadata')
    return np.stack([frames[i] for i in wanted]), {v:i for i,v in enumerate(wanted)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id',required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    start = time.time()
    out = ROOT/'artifacts/slret_goal_v2'/args.run_id
    source = ROOT/'artifacts/slret_goal_v2/external/SignRep'
    checkpoint = source/'ckpt.pt'
    plan = make_plan(args.smoke)
    contract = dict(plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
        code_sha256={str(p):sha(p) for p in (Path(__file__),ROOT/'methods/seds_adaptation/geometry_cache.py')},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        source_sha256={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py'))},
        checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
        precision='FP32 model and input; FP16 cache',batch_size=8,
        preprocessing='RGB cv2.INTER_LINEAR224, ImageNet mean/std, native exact16 frame IDs')
    old = None
    if args.resume:
        old = json.loads((out/'run.json').read_text())
        if old['status']=='completed' or old['contract']!=contract:raise ValueError('Invalid resume contract')
        shutil.copy2(out/'run.json',out/f'previous_attempt_{len(list(out.glob("previous_attempt_*")))+1:03d}.json')
    else:
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'plan.json',plan)
    report = dict(run_id=args.run_id,status='running',pid=os.getpid(),start_unix=start,
        contract=contract,total_videos=len(plan['items']),total_clips=plan['total_clips'],
        completed=0,completed_clips=0,resumed=0,test_used=False,retrieval_training=False,
        smoke=args.smoke,prior_wall_seconds=old.get('total_wall_seconds',old['wall_seconds']) if old else 0.)
    def record():
        report['wall_seconds']=time.time()-start
        report['total_wall_seconds']=report['prior_wall_seconds']+report['wall_seconds']
        atomic_json(out/'run.json',report)
    def terminate(signum,frame):raise TimeoutError('Hard timeout; complete per-video caches retained')
    signal.signal(signal.SIGTERM,terminate)
    record()
    try:
        import cv2
        import numpy as np
        import torch
        torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(42)
        used=sum(x.stat().st_size for x in out.parent.rglob('*') if x.is_file())
        reserve=256*1024**2
        if used+reserve>36*1024**3 or shutil.disk_usage(out).free-reserve<15*1024**3:
            raise ValueError('Storage admission failed')
        torch.cuda.set_per_process_memory_fraction(.25)
        model=load_model(source,checkpoint)
        report.update(strict_checkpoint_loaded=True,torch=torch.__version__,gpu=torch.cuda.get_device_name())
        mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None,None]
        std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None,None]
        torch.cuda.reset_peak_memory_stats()
        for item in plan['items']:
            dest=out/item['split']/(item['id']+'.npz');dest.parent.mkdir(exist_ok=True)
            raw=Path(item['raw_path']).stat()
            if (raw.st_size,raw.st_mtime_ns)!=(item['raw_bytes'],item['raw_mtime_ns']):
                raise ValueError('Raw video changed')
            n=len(item['windows']);tick=time.time()
            if dest.exists():
                with np.load(dest,allow_pickle=False) as cache:
                    for key in ('features','latent'):
                        if cache[key].shape!=(n,768) or not np.isfinite(cache[key]).all():raise ValueError('Invalid prior cache')
                    if not np.array_equal(cache['windows'],item['windows']):raise ValueError('Prior cache alignment differs')
                report['resumed']+=1
            else:
                frames,positions=decode(item)
                collected={k:[] for k in ('features','latent')}
                for offset in range(0,n,8):
                    indices=np.array([[positions[i] for i in w] for w in item['windows'][offset:offset+8]])
                    clips=torch.from_numpy(frames[indices]).permute(0,4,1,2,3).to('cuda',dtype=torch.float32)/255.
                    clips=(clips-mean)/std
                    with torch.inference_mode():result=model(clips)
                    for key in collected:
                        value=result[key]
                        if value.shape!=(len(indices),768) or not torch.isfinite(value).all():raise ValueError('Invalid SignRep feature')
                        if not (value.float().norm(dim=-1)>0).all():raise ValueError('Degenerate SignRep features')
                        collected[key].append(value.cpu().numpy().astype(np.float16))
                    del result,clips
                values={k:np.concatenate(v) for k,v in collected.items()}
                tmp=dest.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez_compressed(f,**values,windows=np.asarray(item['windows']),clip_starts=np.asarray(item['clip_starts']))
                os.replace(tmp,dest)
            report['completed']+=1;report['completed_clips']+=n
            report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
            record()
            print(json.dumps(dict(event='video',id=item['id'],split=item['split'],clips=n,
                completed=report['completed'],total=report['total_videos'],seconds=time.time()-tick)),flush=True)
        report.update(status='completed',exit_status=0)
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc())
        raise
    finally:record()


if __name__=='__main__':main()
