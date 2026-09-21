"""Bounded C09 geometry ETL: fixed TRAIN512 + DEV519, exact native clip centers.

Run once in background; no follow-on training. Completed per-video caches are
retained on failure/timeout and can be explicitly resumed with identical code.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from methods.seds_adaptation.geometry_cache import selected_metadata, clip_centers, SELECTION_SALT
from methods.seds_adaptation.pose3d_branch import joint_indices, JOINT_NAMES
from research.slret_goal_v2.tools.extract_h4w_sample import (
    person_box, projection_to_pixels, set_inference_flags, validate_hand_detector)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def make_plan():
    items=[]
    for split,count in [('train',512),('dev',519)]:
        root = ROOT/f'artifacts/slret_goal/seds-adapted-{split}-001'
        for path in selected_metadata(root/'metadata',count):
            metadata = json.loads(path.read_text())
            centers,starts = clip_centers(metadata)
            raw = Path(metadata['raw_path'])
            if metadata['id'] != path.stem or raw.parent.name != split or not raw.is_file():
                raise ValueError('Video identity/split/source mismatch')
            stat=raw.stat()
            items.append(dict(id=path.stem,split=split,raw_path=str(raw),
                raw_bytes=stat.st_size,raw_mtime_ns=stat.st_mtime_ns,
                metadata_path=str(path),metadata_sha256=digest(path),
                decoded_frames=metadata['decoded_frames'],clip_starts=starts.tolist(),
                clip_frame_ids=centers.tolist(),frame_ids=sorted(set(centers.tolist()))))
    return dict(schema=1,selection_salt=SELECTION_SALT,train_count=512,dev_count=519,
                test_used=False,sampling='native16frame_window_offset8_no_interpolation',items=items,
                total_frames=sum(len(x['frame_ids']) for x in items))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--source',default='/home/haipd/DexAvatar/Hand4Whole-plus-plus_RELEASE')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    started=time.time()
    plan=make_plan()
    if args.plan_only:
        print(json.dumps({k:v for k,v in plan.items() if k!='items'},indent=2));return
    out=ROOT/'artifacts/slret_goal_v2'/args.run_id
    source=Path(args.source)
    code=[Path(__file__),ROOT/'methods/seds_adaptation/geometry_cache.py',
          ROOT/'methods/seds_adaptation/pose3d_branch.py',
          ROOT/'research/slret_goal_v2/tools/extract_h4w_sample.py',
          source/'main/model.py',source/'main/config.py',source/'common/utils/smpl_x.py']
    packages={n:importlib.metadata.version(n) for n in
              ('torch','torchvision','numpy','smplx','timm','ultralytics','mmcv','kornia')}
    checkpoint=source/'output/model_dump/snapshot_6.pth'
    contract=dict(plan_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),
                  source=str(source),code_sha256={str(p):digest(p) for p in code},packages=packages,
                  checkpoint=dict(path=str(checkpoint),bytes=checkpoint.stat().st_size,
                                  mtime_ns=checkpoint.stat().st_mtime_ns),crop='native_person_1.25')
    previous=None
    if args.resume:
        previous=json.loads((out/'run.json').read_text())
        if previous['status']=='completed' or previous['contract']!=contract:
            raise ValueError('Completed run or changed resume contract')
        attempt=len(list(out.glob('previous_attempt_*.json')))+1
        shutil.copy2(out/'run.json',out/f'previous_attempt_{attempt:03d}.json')
    else:
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'plan.json',plan)
    prior_wall=previous.get('total_wall_seconds',previous['wall_seconds']) if previous else 0.
    report=dict(status='running',run_id=args.run_id,pid=os.getpid(),start_unix=started,
                contract=contract,completed=0,resumed=0,completed_frames=0,new_frames=0,
                total_videos=len(plan['items']),total_frames=plan['total_frames'],
                test_used=False,retrieval_training=False,interpreter=sys.executable,
                coordinate_system='estimated pelvis-relative camera-axis XYZ; original-pixel UV',
                output_bytes=0,inference_seconds=0.,detection_seconds=0.)
    def record():
        report['wall_seconds']=time.time()-started
        report['total_wall_seconds']=prior_wall+report['wall_seconds']
        atomic_json(out/'run.json',report)
    def terminate(signum,frame):
        raise TimeoutError('Bounded launcher sent SIGTERM; completed videos retained')
    signal.signal(signal.SIGTERM,terminate)
    record()
    try:
        import cv2
        import numpy as np
        import torch
        torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
        if shutil.disk_usage(out).free < 16*1024**3:
            raise RuntimeError('15GiB disk reserve plus1GiB headroom required')
        sys.path[:0]=[str(source/'main'),str(source/'common')]
        os.environ.setdefault('PYOPENGL_PLATFORM','egl')
        from config import cfg
        from model import get_model
        from utils.preprocessing import set_aspect_ratio,get_patch_img
        from utils.smpl_x import smpl_x
        from ultralytics import YOLO
        model=get_model('test').cuda()
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state={k.removeprefix('module.'):v for k,v in saved['network'].items()}
        loaded=model.load_state_dict(state,strict=False)
        external=('wilor.','dwpose.','wilor_det.','smplx_layer.')
        missing_core=[k for k in loaded.missing_keys if not k.startswith(external)]
        report['load_keys']=dict(missing=loaded.missing_keys,unexpected=loaded.unexpected_keys,
                                missing_core=missing_core)
        if missing_core or loaded.unexpected_keys:
            raise ValueError('Unexplained checkpoint key mismatch')
        del saved,state
        set_inference_flags(model)
        report['hand_detector_invariant']=validate_hand_detector(model)
        detector_path=source/'demo/yolo11n.pt'
        if not detector_path.is_file():raise FileNotFoundError(detector_path)
        detector=YOLO(str(detector_path))
        indices=joint_indices(list(smpl_x.kpt['name']))
        torch.cuda.reset_peak_memory_stats();record()
        for item in plan['items']:
            video_start=time.time()
            folder=out/item['split'];folder.mkdir(exist_ok=True)
            target=folder/(item['id']+'.npz')
            if args.resume and target.exists():
                with np.load(target,allow_pickle=False) as cached:
                    if (not np.array_equal(cached['frame_ids'],item['frame_ids'])
                            or not np.array_equal(cached['clip_frame_ids'],item['clip_frame_ids'])
                            or not np.array_equal(cached['clip_starts'],item['clip_starts'])
                            or cached['xyz'].shape!=(len(item['frame_ids']),49,3)
                            or not np.isfinite(cached['xyz']).all()
                            or list(cached['joint_names'])!=list(JOINT_NAMES)):
                        raise ValueError('Invalid completed video on resume')
                report['resumed']+=1
            else:
                if target.exists():raise FileExistsError(target)
                if shutil.disk_usage(out).free < 15*1024**3:
                    raise RuntimeError('15GiB reserve reached')
                stat=Path(item['raw_path']).stat()
                if (stat.st_size,stat.st_mtime_ns)!=(item['raw_bytes'],item['raw_mtime_ns']):
                    raise ValueError('Source video changed after plan')
                cap=cv2.VideoCapture(item['raw_path'])
                wanted=set(item['frame_ids']);frame_id=0
                xyzs=[];uvs=[];ids=[];boxes=[];fallbacks=[];inside=[]
                try:
                    while True:
                        ok,bgr=cap.read()
                        if not ok:break
                        current=frame_id;frame_id+=1
                        if current not in wanted:continue
                        if bgr.shape!=(260,210,3):raise ValueError('Unexpected PH dimensions')
                        h,w=bgr.shape[:2]
                        torch.cuda.synchronize();tick=time.time()
                        detections=detector(bgr,verbose=False,device=0)[0].boxes
                        torch.cuda.synchronize();report['detection_seconds']+=time.time()-tick
                        raw_box,fallback=person_box(detections,w,h)
                        bbox=set_aspect_ratio(raw_box,cfg.input_img_shape[1]/cfg.input_img_shape[0])
                        crop,_,inverse=get_patch_img(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),bbox,
                                                    1.,0.,False,cfg.input_img_shape)
                        tensor=torch.from_numpy(crop.astype(np.float32).transpose(2,0,1)).contiguous().cuda()[None]/255.
                        torch.cuda.synchronize();tick=time.time()
                        with torch.inference_mode():result=model({'img':tensor},{},{},'test')
                        torch.cuda.synchronize();report['inference_seconds']+=time.time()-tick
                        xyz=result['smplx_kpt_cam'][0,indices].float().cpu().numpy()
                        uv=projection_to_pixels(result['smplx_kpt_proj'][0,indices].float().cpu().numpy(),
                                                cfg.input_img_shape,cfg.vit_output_shape,inverse)
                        if not np.isfinite(xyz).all() or not np.isfinite(uv).all():
                            raise ValueError('Nonfinite geometry; no silent frame dropping')
                        xyzs.append(xyz);uvs.append(uv);ids.append(current);boxes.append(bbox);fallbacks.append(fallback)
                        inside.append(float(((uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)).mean()))
                        report['new_frames']+=1
                finally:
                    cap.release()
                if ids!=item['frame_ids'] or frame_id!=item['decoded_frames']:
                    raise ValueError('Raw decode/native temporal alignment mismatch')
                # Per-video atomic output; preserve interrupted temp files, never overwrite history.
                temporary=folder/(item['id']+f'.{os.getpid()}.tmp')
                with temporary.open('xb') as handle:
                    np.savez_compressed(handle,xyz=np.stack(xyzs),uv=np.stack(uvs),frame_ids=np.asarray(ids),
                        clip_frame_ids=np.asarray(item['clip_frame_ids']),clip_starts=np.asarray(item['clip_starts']),
                        joint_names=np.asarray(JOINT_NAMES),crop_boxes=np.stack(boxes),
                        person_fallback=np.asarray(fallbacks),projected_inside_fraction=np.asarray(inside))
                os.replace(temporary,target)
            report['completed']+=1;report['completed_frames']+=len(item['frame_ids'])
            report['output_bytes']+=target.stat().st_size
            report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
            record()
            print(json.dumps(dict(event='video_complete',split=item['split'],id=item['id'],
                completed=report['completed'],total=report['total_videos'],frames=report['completed_frames'],
                total_frames=report['total_frames'],seconds=time.time()-video_start,
                wall_seconds=report['wall_seconds'])),flush=True)
        report.update(status='completed',exit_status=0,decision='await_user_return_for_matched_XY_XYZ_training')
    except Exception:
        report.update(status='failed',exit_status=1,error=traceback.format_exc());raise
    finally:
        record()


if __name__=='__main__':main()
