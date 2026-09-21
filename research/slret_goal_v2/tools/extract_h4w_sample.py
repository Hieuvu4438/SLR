"""Bounded TRAIN-only H4W feasibility sample; no retrieval/extraction queue."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def person_box(boxes, width, height):
    """Native demo policy: highest-confidence person, explicit full-frame fallback."""
    people=[box for box in boxes if int(box.cls.item()) == 0]
    if not people:
        return [0.,0.,float(width),float(height)], True
    box=max(people,key=lambda b:float(b.conf.item()))
    x1,y1,x2,y2=box.xyxy[0].tolist()
    if x2<=x1 or y2<=y1:
        raise ValueError('Degenerate person detection')
    return [x1,y1,x2-x1,y2-y1], False


def projection_to_pixels(uv, crop_shape, output_shape, inverse_affine):
    import numpy as np
    uv=np.asarray(uv,dtype=np.float32).copy()
    uv*=np.array([crop_shape[1]/output_shape[1],crop_shape[0]/output_shape[0]],np.float32)
    return np.concatenate([uv,np.ones((len(uv),1),np.float32)],axis=1)@np.asarray(inverse_affine).T


def set_inference_flags(model):
    """Never recurse through YOLO.train(): it launches a training workflow.

    nn.Module.eval() calls train(False) recursively, but Ultralytics' wrapper
    overloads train with different semantics. Explicit flags preserve the
    ordinary BN/dropout inference behavior without calling wrapper workflows.
    """
    for module in model.modules():
        module.training=False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(module.training for module in model.modules()):
        raise ValueError('Inference flag invariant failed')


def validate_hand_detector(model):
    detector=model.wilor_det.detector
    head=detector.model.model[-1]
    shape=list(head.kpt_shape)
    names=dict(detector.model.names)
    if len(names)!=2 or shape!=[21,3] or getattr(detector,'trainer',None) is not None:
        raise ValueError('Expected untouched two-hand/21-keypoint detector, no trainer')
    return dict(names=names,keypoint_shape=shape,trainer_absent=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--source',default='/home/haipd/DexAvatar/Hand4Whole-plus-plus_RELEASE')
    parser.add_argument('--crop-policy',choices=['full','person'],default='full')
    args=parser.parse_args()
    out=ROOT/'artifacts/slret_goal_v2'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    report=dict(status='running',run_id=args.run_id,pid=os.getpid(),split='train',test_used=False,frames=[],
                sample_policy='first3 canonical TRAIN IDs, centered8 consecutive decoded frames each',
                retrieval_training=False,source=args.source,crop_policy=args.crop_policy,
                interpreter=sys.executable,seed=42,
                extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def record():
        report['wall_seconds']=time.time()-started
        (out/'run.tmp').write_text(json.dumps(report,indent=2,allow_nan=False))
        os.replace(out/'run.tmp',out/'run.json')
    record()
    def on_timeout(signum, frame):
        raise TimeoutError('Bounded launcher sent SIGTERM')
    signal.signal(signal.SIGTERM,on_timeout)
    try:
        import cv2
        import numpy as np
        import torch
        import importlib.metadata
        torch.manual_seed(42); np.random.seed(42)
        torch.set_num_threads(4)
        report['packages']={name:importlib.metadata.version(name) for name in
                            ('torch','torchvision','numpy','smplx','timm','ultralytics','mmcv','kornia')}
        from methods.seds_adaptation.pose3d_branch import joint_indices, JOINT_NAMES
        source=Path(args.source)
        sys.path[:0]=[str(source/'main'),str(source/'common')]
        os.environ.setdefault('PYOPENGL_PLATFORM','egl')
        from config import cfg
        from model import get_model
        from utils.preprocessing import set_aspect_ratio,get_patch_img
        from utils.smpl_x import smpl_x
        ckpt=source/'output/model_dump/snapshot_6.pth'
        report['checkpoint']=dict(path=str(ckpt),bytes=ckpt.stat().st_size)
        report['source_sha256']={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [source/'main/model.py',source/'main/config.py',source/'common/utils/smpl_x.py']}
        model=get_model('test').cuda()
        saved=torch.load(ckpt,map_location='cpu',weights_only=False)
        state=saved['network']
        state={k.removeprefix('module.'):v for k,v in state.items()}
        result=model.load_state_dict(state,strict=False)
        # WiLoR/DWPose load their own local weights in constructors; SMPL-X is
        # a separately licensed local model. Never allow missing core weights.
        external=('wilor.','dwpose.','wilor_det.','smplx_layer.')
        missing_core=[k for k in result.missing_keys if not k.startswith(external)]
        report['load_keys']=dict(missing=result.missing_keys,unexpected=result.unexpected_keys,
                                 missing_core=missing_core,external_prefixes=list(external))
        record()
        if missing_core or result.unexpected_keys:
            raise ValueError('Unexplained checkpoint key mismatch; no inference')
        del saved,state
        set_inference_flags(model)
        report['hand_detector_invariant']=validate_hand_detector(model)
        detector=None
        if args.crop_policy=='person':
            from ultralytics import YOLO
            detector_path=source/'demo/yolo11n.pt'
            if not detector_path.is_file():
                raise FileNotFoundError(detector_path)
            detector=YOLO(str(detector_path))
            report['person_detector']=dict(path=str(detector_path),bytes=detector_path.stat().st_size)
        index=joint_indices(list(smpl_x.kpt['name']))
        rows=[json.loads(line) for line in (ROOT/'artifacts/manifests/ph_train.jsonl').read_text().splitlines()][:3]
        report['ids']=[r['video_id'] for r in rows]
        report['joint_names']=list(JOINT_NAMES)
        report['coordinate_system']='pelvis-relative camera-axis XYZ; original-pixel projection stored separately'
        record()
        torch.cuda.reset_peak_memory_stats()
        for row in rows:
            metadata=json.loads(Path(row['temporal_metadata']).read_text())
            video=Path(metadata['source_video'])
            if '/train/' not in str(video) or not video.is_file():
                raise ValueError('Expected existing TRAIN video')
            count=metadata['decoded_frame_count']
            first=max(0,count//2-4); wanted=set(range(first,min(count,first+8)))
            cap=cv2.VideoCapture(str(video)); frames=[]; frame_id=0
            while cap.isOpened() and frame_id<=max(wanted):
                ok,bgr=cap.read()
                if not ok: break
                if frame_id in wanted: frames.append((frame_id,bgr))
                frame_id+=1
            cap.release()
            if len(frames)!=len(wanted): raise ValueError('Decode/frame manifest mismatch')
            xyz_all=[]; uv_all=[]; ids=[]
            for frame_id,bgr in frames:
                h,w=bgr.shape[:2]
                raw_bbox=[0.,0.,float(w),float(h)]; fallback=False
                detection_seconds=0.
                if detector is not None:
                    torch.cuda.synchronize(); detection_start=time.time()
                    boxes=detector(bgr,verbose=False,device=0)[0].boxes
                    torch.cuda.synchronize(); detection_seconds=time.time()-detection_start
                    raw_bbox,fallback=person_box(boxes,w,h)
                bbox=set_aspect_ratio(raw_bbox,cfg.input_img_shape[1]/cfg.input_img_shape[0])
                crop,img2bb,bb2img=get_patch_img(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),bbox,1.,0.,False,cfg.input_img_shape)
                tensor=torch.from_numpy(crop.astype(np.float32).transpose(2,0,1)).contiguous().cuda()[None]/255.
                torch.cuda.synchronize(); tick=time.time()
                with torch.inference_mode(): result=model({'img':tensor},{},{},'test')
                torch.cuda.synchronize(); elapsed=time.time()-tick
                xyz=result['smplx_kpt_cam'][0,index].float().cpu().numpy()
                uv=result['smplx_kpt_proj'][0,index].float().cpu().numpy()
                uv=projection_to_pixels(uv,cfg.input_img_shape,cfg.vit_output_shape,bb2img)
                if not np.isfinite(xyz).all() or not np.isfinite(uv).all(): raise ValueError('Nonfinite output')
                xyz_all.append(xyz); uv_all.append(uv); ids.append(frame_id)
                report['frames'].append(dict(video_id=row['video_id'],frame_id=frame_id,seconds=elapsed,
                    detection_seconds=detection_seconds,person_fallback=fallback,
                    raw_bbox=raw_bbox,crop_bbox=bbox.tolist(),
                    z_span=float(np.ptp(xyz[:,2])),projected_inside_fraction=float(((uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)).mean())))
                if len(ids)==1:
                    cv2.imwrite(str(out/(row['video_id']+'_input.jpg')),bgr)
                    overlay=bgr.copy()
                    for j,(u,v) in enumerate(uv):
                        if abs(u)<10000 and abs(v)<10000:
                            cv2.circle(overlay,(round(float(u)),round(float(v))),2,[(0,255,0),(0,0,255),(255,255,0)][min(j//21,2)],-1)
                    cv2.imwrite(str(out/(row['video_id']+'_projection.jpg')),overlay)
                print(json.dumps(report['frames'][-1]),flush=True); record()
            np.savez_compressed(out/(row['video_id']+'.npz'),xyz=np.stack(xyz_all),uv=np.stack(uv_all),
                                frame_ids=np.asarray(ids),joint_names=np.asarray(JOINT_NAMES))
        report.update(status='completed',peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      inference_seconds=sum(x['seconds'] for x in report['frames']),
                      detection_seconds=sum(x['detection_seconds'] for x in report['frames']),
                      decision='await_sample_quality_and_cost_review_before_full_extraction')
    except Exception:
        report.update(status='failed',error=traceback.format_exc()); raise
    finally:
        record()


if __name__=='__main__': main()
