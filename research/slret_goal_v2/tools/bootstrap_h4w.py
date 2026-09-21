"""User-authorized isolated environment rebuild, then bounded 24-frame probe.

No existing environment/source is modified; refuse an existing target prefix.
No retries or corpus extraction. Outer launch_bounded enforces process timeout.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
TARGET=Path('/home/haipd/miniconda3/envs/h4wpp')
SOURCE=Path('/home/haipd/miniconda3/envs/dexavatar')
OUT=ROOT/'artifacts/slret_goal_v2/h4w-runtime-002'


def main():
    if TARGET.exists(): raise FileExistsError('Refuse overwriting an existing environment')
    if shutil.disk_usage(ROOT).free<40*1024**3: raise RuntimeError('Need40GiB free before isolated clone')
    OUT.mkdir(parents=True,exist_ok=False)
    started=time.time()
    report=dict(status='running',pid=os.getpid(),target=str(TARGET),clone_source=str(SOURCE),steps=[],
                authorization='user explicitly requested rebuilding deleted h4wpp and downloading packages',test_used=False)
    def record():
        report['wall_seconds']=time.time()-started
        (OUT/'run.tmp').write_text(json.dumps(report,indent=2)); os.replace(OUT/'run.tmp',OUT/'run.json')
    def run(name,command):
        report['current']=dict(name=name,command=command); record()
        print(json.dumps(report['current']),flush=True)
        tick=time.time()
        env=os.environ.copy(); env.update(PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONNOUSERSITE='1')
        # Conda/pip may replace hard-linked files during installation. Conda
        # --copy below guarantees the source environment is never shared-writable.
        result=subprocess.run(command,cwd=ROOT,env=env)
        report['steps'].append(dict(name=name,returncode=result.returncode,seconds=time.time()-tick)); record()
        if result.returncode: raise RuntimeError(f'{name} failed; no retry')
        if shutil.disk_usage(ROOT).free<15*1024**3: raise RuntimeError('Disk reserve reached; stop')
    try:
        run('clone',['/home/haipd/miniconda3/bin/conda','create','--yes','--copy','--prefix',str(TARGET),'--clone',str(SOURCE)])
        py=str(TARGET/'bin/python')
        # Pins mirror the saved successful h4wpp environment; preserve its
        # torch2.1.1/cu121 and working PyTorch3D0.7.5 inherited from dexavatar.
        run('packages',[py,'-m','pip','install','--no-cache-dir','numpy==1.26.3','torch==2.1.1+cu121',
            'torchvision==0.16.1+cu121','setuptools==69.5.1','opencv-python==4.11.0.86',
            'ultralytics==8.4.126','mmengine==0.10.7','mmdet==3.3.0','xtcocotools==1.14.3',
            'json-tricks==3.17.3','munkres==1.1.4'])
        run('mmcv-wheel',[py,'-m','pip','install','--no-cache-dir','--no-deps',
            'https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl'])
        run('imports',[py,'-c',
            'import torch,torchvision,mmcv,mmengine,ultralytics; from mmcv.ops import nms; from pytorch3d.ops import corresponding_points_alignment; print(torch.__version__,torchvision.__version__,mmcv.__version__,mmengine.__version__,ultralytics.__version__); assert torch.__version__=="2.1.1+cu121"'])
        # pip check is diagnostic: cloned non-H4W packages may have pre-existing
        # incompatibilities; real H4W imports/inference below must succeed.
        with (OUT/'pip-check.txt').open('w') as handle:
            checked=subprocess.run([py,'-m','pip','check'],stdout=handle,stderr=subprocess.STDOUT)
        report['pip_check_exit']=checked.returncode
        with (OUT/'pip-freeze.txt').open('w') as handle:
            subprocess.run([py,'-m','pip','freeze'],stdout=handle,check=True)
        run('train-only-sample',[py,str(ROOT/'research/slret_goal_v2/tools/extract_h4w_sample.py'),
                                 '--run-id','h4w-ph-train-sample-001'])
        report.update(status='completed',current=None)
    except Exception:
        report.update(status='failed',error=traceback.format_exc()); raise
    finally:
        record()


if __name__=='__main__': main()
