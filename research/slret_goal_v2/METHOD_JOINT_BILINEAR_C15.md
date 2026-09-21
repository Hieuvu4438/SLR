# C15 — within-part joint bilinear representation

## Material Passport

academic-research-suite / experiment-agent, inline implementation/run; 2026-09-20.
User approved another7200s after C14-R1, explicitly moving beyond C13/C14 tuning.

- Hypothesis: native independent channel maxima over finger/arm joints discard
  local channel coactivation; retaining second-order statistics may help fine-grained
  pose discrimination. Source `modeling_gcn.py:forward`, before first anatomical pool.
- Add per-joint nonaffine channel LayerNorm → bias-free256→16 → masked mean outer
  product → upper triangle136 → smooth signed-sqrt/L2 → zero-init136→256 residual
  onto each native part maximum. Same sharedhand module for left/right, separatebody.
  38912parameters/encoder,77824total. Raw XY validity excludes missing joints only
  in the added descriptor; original pooling is unchanged. No confidence inferred.
- RGB and original pose2D remain. No graph/bone/3D/temporal/scorer/loss change.
- Borrow outer-product pooling from [Lin et al., ICCV2015](https://openaccess.thecvf.com/content_iccv_2015/html/Lin_Bilinear_CNN_Models_ICCV_2015_paper.html).
  Primary official abstract verified; not a full-paper reproduction or novelty claim.
- Collision: C03 multiplied already-pooled left/right/body vectors; C15 aggregates
  within-part joint-level channel products BEFORE information is discarded, while
  adapting native GCN. Unlike C07 it has no QKV joint exchange; unlike C13/C14 it
  changes neither adjacency nor raw descriptors. Not frozen sentence/clip weighting,
  local rival readout or any user-closed C06 temporal delta; NO_GO_REGISTRY checked.
- Release warmstart, allGCN+fusion plus newmodule; seed42/B32/3epochs666updates,
  GCN1e-6/fusion1e-5/bilinear1e-4, native seven losses, FP32moments, nativeGCN BN
  and dropout fixed eval. Fresh online pose, cached upstream RGB; no stale posecache.
- Smoke3updates checks zero-init fullDEV parity once, both input/output projections
  update, frozen tensors/buffers preserved. Pilot inherits exact smoke source/base/data.
- Full DEV519 at111/222/444/666; mean bidirectional R1, init eligible, earliest ties,
  each direction >=release-.5pp; registered earlystop if mean<release-2pp. No TEST.
- Compare release77.552987, historical matched R1seed4278.612717, best133778.709056.
  R2 replay mismatch remains unresolved; historical contrast exploratory. If lead,
  matched first-order equal-budget descriptor control then seeds before mechanismclaim.
  If flat/degraded, inspect curve and activation once; no blind LR/rank sweep.
- Expected1800–2400s,<=24GiBVRAM,3GiBpilot+128MiBsmoke; caps120/2500s,
  parent2660s reserved of8031.479704s available. Storage cap36GiB/freefloor15GiB.
- Retire only C14-R1last and C04seed2026last (1.902939GiB); preserve selectedbest,
  allGCN/checkpoints in use/data/features/logs. Permanent deletion manifest recorded.
- Command: `launch_bounded.py --name v2-c15-joint-bilinear-pilot-001 --seconds 2660
  -- /home/haipd/miniconda3/envs/seds/bin/python research/slret_goal_v2/tools/run_adaptive_graph.py --joint-bilinear`.
  Outputs `artifacts/slret_goal_v2/c15-joint-bilinear-pilot-001/`, child
  `seds-joint-bilinear-001/`. Background then yield; no polling/retry/follow-on queue.

Scoped CPU checks:13 passed (C15/masked-recipe/bone regression only); not evidence
of retrieval gain. GPU smoke and pilot pending at registration.

Launched2026-09-20 14:56:55 local (unix1789891015.6609783), parentPID3272816,
timeoutPID3272815, hard2660s. Cleanup completed:2,043,265,366bytes permanently
removed; manifest `artifacts/slret_goal_v2/storage-prune-c15-20260920-001/manifest.json`.
Persistent parent log: `artifacts/slret_goal/jobs/v2-c15-joint-bilinear-pilot-001/console.log`;
smoke/pilot logs under the C15 chain directory. Collection/accounting pending.

## Implementation-gate repair before pilot

Chain001 FAILED, smoke00125.119404554367065s, only one recorded step; no longpilot.
FullDEV zero-init parity passed. Gate incorrectly required input projection update2:
native BertAdam starts with LR0 at update1; zerooutput first moves at update2,
so task gradient into input projection can first flow at backward3. Tiny CPU test
with the actual native optimizer reproduces this chronology (weight_decay0).
Repair requires output update2, finite nonzero input gradient AND both projection
updates at3. No relaxation of the eventual activation requirement. Smoke now3updates;
pilot remains666 with unchanged model/LRs/schedule/data. Four scoped C15 tests pass.
Original failed source/logs retained; no efficacy conclusion from invalid smoke.
Retry is a deliberate implementation repair under goal §7, not driver autoretry.
New smokeID `seds-joint-bilinear-smoke-002`, chain `c15-joint-bilinear-pilot-002`,
launchername `v2-c15-joint-bilinear-pilot-002`; original unused longpilotID unchanged.
Charge25.119405s once, exclude31.179850s parent; available8006.360300s, reserve2660s.

Smoke002 COMPLETED3updates/30.920886754989624s: exact registered zero-init score
parity tolerance, output/input activation and nonzero input-gradient checks pass;
nativeGCN/fusion updates and frozen-buffer checks pass. ParentPID3276752 alive,
now running `seds-joint-bilinear-001` (child timeoutPID3277643). Chain002 start
unix1789891201.0068195. No more source edits while live. Successful smoke runtime
remains inside2660s reservation and will be charged with pilot on collection,
not charged yet; do not count it twice. No long-training monitoring loop.

## Collected result

Completed666/allgatespass/noearlystop. Selected666mean78.4200385356,
T77.4566473988/V79.3834296724, -.1926782274ppR1seed42/-.2890173410ppglobalbest.
Curves111/222/444/666=77.938343/78.131021/78.227360/78.420039; no promotion.
Bestsha93518a5d2781ef84b84a14402b069c136287bf34787dc1bece925c7950f719e6 kept.
Charge30.920887+1771.452873=1802.373760s once; failedsmoke already accounted.
One structural refinement registered in METHOD_JOINT_COVARIANCE_C15_R1.md.
