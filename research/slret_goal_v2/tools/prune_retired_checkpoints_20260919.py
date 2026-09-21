"""One-time user-authorized cleanup, fixed allowlist; retain reports and models in use."""
import argparse
import json
import os
from pathlib import Path
import stat
import time

ROOT=Path(__file__).resolve().parents[3]
TARGETS={
    'seds-aux025-001':['last.pt'],
    'seds-staged-001':['last.pt'],
    'seds-articulator-001':['last.pt'],
    'seds-articulator-frozen-001':['last.pt'],
    'seds-smooth005-001':['last.pt'],
    'seds-joint-cross-001':['last.pt'],
    'seds-joint-cross-lrfix-001':['last.pt'],  # Keep selected best.pt.
    'seds-lora-b128-001':['best.pt','last.pt'],
    'seds-fusion-b128-001':['best.pt','last.pt'],
    'seds-lora-depth3-001':['best.pt','last.pt'],
    'seds-lora-text-001':['best.pt','last.pt'],
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    parser.add_argument('--c12-last-only',action='store_true',
                        help='Separate one-file cleanup; preserve C12 best.pt and all provenance')
    parser.add_argument('--c11-last-only',action='store_true',
                        help='Separate one-file cleanup; preserve C11 best.pt and all provenance')
    parser.add_argument('--gcn-replication-space',action='store_true',
                        help='Retire four completed unused last states; preserve all selected checkpoints')
    parser.add_argument('--gcn-freeze-space',action='store_true')
    parser.add_argument('--adaptive-graph-space',action='store_true')
    parser.add_argument('--c13-last-only',action='store_true')
    parser.add_argument('--c14-space',action='store_true')
    parser.add_argument('--c14-refinement-space',action='store_true')
    parser.add_argument('--c15-space',action='store_true')
    parser.add_argument('--c15-refinement-space',action='store_true')
    parser.add_argument('--c16-space',action='store_true')
    args=parser.parse_args()
    if sum((args.c16_space,args.c12_last_only,args.c11_last_only,args.gcn_replication_space,args.gcn_freeze_space,args.adaptive_graph_space,args.c13_last_only,args.c14_space,args.c14_refinement_space,args.c15_space,args.c15_refinement_space))>1:raise ValueError('Choose one cleanup')
    base=ROOT/'artifacts/slret_goal_v2'
    entries=[]
    targets={'seds-gcn-clip-temporal-001':['last.pt']} if args.c12_last_only else TARGETS
    if args.c11_last_only:targets={'seds-gcn-lora-001':['last.pt']}
    if args.c13_last_only:targets={'seds-adaptive-graph-001':['last.pt']}
    if args.c14_space:targets={name:['last.pt'] for name in ('seds-adaptive-graph-lr1e5-001','seds-lora-001')}
    if args.c14_refinement_space:targets={name:['last.pt'] for name in ('seds-bone-features-001','seds-lora-seed1337-001')}
    if args.c15_space:targets={name:['last.pt'] for name in ('seds-bone-features-lr1e4-001','seds-lora-seed2026-001')}
    if args.c15_refinement_space:targets={name:['last.pt'] for name in ('seds-joint-bilinear-001','seds-masked-control-002')}
    if args.c16_space:targets={name:['last.pt'] for name in ('seds-joint-covariance-001','seds-gcn-clean-seed2026-001')}
    if args.gcn_freeze_space:
        targets={name:['last.pt'] for name in ('seds-lora-control-fusion-001','seds-fusion-seed1337-001')}
    if args.adaptive_graph_space:
        targets={name:['last.pt'] for name in ('seds-gcn-freeze-001','seds-fusion-seed2026-001',
                                             'seds-gcn-fusion-ablation-seed1337-001')}
    if args.gcn_replication_space:
        targets={name:['last.pt'] for name in ('seds-masked-pose-002',
            'seds-gcn-horizon3-fusion-control-001','seds-gcn-fusion-ablation-seed42-001',
            'seds-gcn-fusion-ablation-seed2026-001')}
    for run,names in targets.items():
        report=json.loads((base/run/'run.json').read_text())
        if report['status']!='completed':raise ValueError(f'Nonterminal run: {run}')
        if args.c16_space or args.gcn_replication_space or args.gcn_freeze_space or args.adaptive_graph_space or args.c13_last_only or args.c14_space or args.c14_refinement_space or args.c15_space or args.c15_refinement_space:
            selected=Path(report['selection']['checkpoint'])
            if not selected.is_file() or selected.resolve()==(base/run/'last.pt').resolve():
                raise ValueError(f'Selected checkpoint must remain available: {run}')
        for name in names:
            path=base/run/name
            info=path.lstat()
            if path.resolve()!=path or not stat.S_ISREG(info.st_mode):
                raise ValueError(f'Not an explicit regular file: {path}')
            entries.append(dict(path=str(path),bytes=info.st_size,inode=info.st_ino,
                                mtime_ns=info.st_mtime_ns,deleted=False))
    manifest=dict(status='planned',authority='User2026-09-19 requests deleting unused old run/method artifacts',
        retained='release checkpoints; C04 incumbent/replications/controls; correctedC07best; all data/features/logs/configs/metrics/code',
        recovery='Permanent file deletion; no trash copy. Recreate weights only by rerunning saved recipe; exact reproducibility not guaranteed.',
        targets=entries,total_bytes=sum(x['bytes'] for x in entries),deleted_bytes=0,start_unix=time.time())
    if args.c12_last_only:
        manifest['retained']='C12 best.pt; all incumbents/seeds; all data/features/logs/configs/metrics/code'
        if not (base/'seds-gcn-clip-temporal-001/best.pt').is_file():
            raise ValueError('C12 best checkpoint must remain available')
    if args.c11_last_only:
        manifest['retained']='C11 best.pt; all incumbents/seeds; all data/features/logs/configs/metrics/code'
        if not (base/'seds-gcn-lora-001/best.pt').is_file():
            raise ValueError('C11 best checkpoint must remain available')
    if args.gcn_replication_space or args.gcn_freeze_space or args.adaptive_graph_space:
        manifest['retained']='All selected checkpoints, GCN/C04 seeds, model/data/features and complete run provenance; only listed unused last states removed'
    print(json.dumps(manifest,indent=2))
    if not args.execute:return
    out=base/('storage-prune-c12-last-20260920-001' if args.c12_last_only else 'storage-prune-20260919-001')
    if args.c11_last_only:out=base/'storage-prune-c11-last-20260920-001'
    if args.gcn_replication_space:out=base/'storage-prune-gcn-replication-20260920-001'
    if args.gcn_freeze_space:out=base/'storage-prune-gcn-freeze-20260920-001'
    if args.adaptive_graph_space:out=base/'storage-prune-adaptive-graph-20260920-001'
    if args.c13_last_only:out=base/'storage-prune-c13-last-20260920-001'
    if args.c14_space:out=base/'storage-prune-c14-20260920-001'
    if args.c14_refinement_space:out=base/'storage-prune-c14-refinement-20260920-001'
    if args.c15_space:out=base/'storage-prune-c15-20260920-001'
    if args.c15_refinement_space:out=base/'storage-prune-c15-refinement-20260920-001'
    if args.c16_space:out=base/'storage-prune-c16-20260920-001'
    out.mkdir(exist_ok=False)
    def record():
        temp=out/'manifest.tmp';temp.write_text(json.dumps(manifest,indent=2)+'\n')
        os.replace(temp,out/'manifest.json')
    record()
    try:
        for entry in entries:
            path=Path(entry['path']);info=path.lstat()
            if (info.st_ino,info.st_size,info.st_mtime_ns)!=(entry['inode'],entry['bytes'],entry['mtime_ns']):
                raise ValueError(f'Target changed: {path}')
            path.unlink()
            entry['deleted']=True;manifest['deleted_bytes']+=entry['bytes'];record()
        manifest['status']='completed'
    except Exception as exc:
        manifest.update(status='failed',error=repr(exc));raise
    finally:
        manifest['wall_seconds']=time.time()-manifest['start_unix'];record()
    print(json.dumps(dict(status=manifest['status'],deleted_files=len(entries),
                         deleted_GiB=manifest['deleted_bytes']/1024**3)))


if __name__=='__main__':main()
