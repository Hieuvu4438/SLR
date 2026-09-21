# C15-R1 — centered within-part covariance

## Material Passport

academic-research-suite / experiment-agent, inline implementation/run2026-09-20.
User reiterates borrowing old/recent papers and extending useful ideas is allowed;
measured accuracy comes first, novelty remains a later evidence-backed objective.

- OriginalC15 completed666: best78.420039 < R1seed4278.612717 < globalbest78.709056.
  It activates and improves overrelease but adds no established gain over GCN-R1.
- Hypothesis: raw second moment includes common-mode outer product of the part
  mean, potentially redundant with retained native max. Center projected joint
  features across valid joints before second-moment pooling to encode differences.
  Algebraic observation, NOT a measured failure attribution or promised improvement.
- Change ONLY `--centered-bilinear`: z←z−masked_mean(z), then mean(zzT).
  Same256→16/136→256,77824params, smooth signed-sqrt/L2, zerooutput, anatomical
  placement and max residual. No new rank/LR/normalizer/temporal branch or loss.
- Covariance pooling prior: [Li et al., CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Towards_Faster_Training_CVPR_2018_paper.html),
  official metadata/abstract verified. PDF retrieval unavailable403; no full-read
  or exact iSQRT-COV reproduction claimed. We do NOT add their matrix-square-root
  algorithm, so centering remains the single experimental contrast.
- Missing-joint mask is from rawXY, not learned confidence. Zero/singleton parts
  yield zero covariance; native branch remains unchanged. Test exact covariance,
  valid-joint common-shift invariance at projected-feature level (NOT rawXY),
  missingness, finite gradients, zero-init identity and checkpoint roundtrip.
- Different from closed score-space centering/whitening and frozen residual probes:
  this modifies trainable per-part joint aggregation before native pooling, no
  gallery/query centering, whitening, teacher, score residual, or relevance change.
- Release/seed42/B32/666updates, nativeGCN1e-6/fusion1e-5/newmodule1e-4,
  native seven losses/FP32moments/fixedGCN BN/dropout. RGB and2D retained, noTEST.
- DEV0/111/222/444/666 with original selector/directional-.5pp guard, stop if
  mean drops>2pp vsrelease; initial eligible, earliest ties. No selector change.
- Compare originalC15 and R1 common endpoints plus selected checkpoints. If above
  R1, replicate then first-order equal-capacity control; if flat/degraded, defer
  family now. No automatic sweep or subsequent experiment while user waits.
- Cap120ssmoke3updates+2500spilot, parent2660s; expected~30min/<=20GiBVRAM.
  Available6203.986540s; reserve2660s/unreserved3543.986540s, no budgetincrease.
- Disk3GiBpilot+128MiBsmoke,36GiBcap/15GiBfreefloor. Retire only unneeded
  C15last and completed one-epochGCNseed42last; both selectedbest remain, as do
  all GCN-R1 states, datasets/features/pretrained and full provenance.
- Command: `launch_bounded.py --name v2-c15-joint-covariance-pilot-001 --seconds
  2660 -- /home/haipd/miniconda3/envs/seds/bin/python research/slret_goal_v2/tools/run_adaptive_graph.py
  --joint-bilinear --centered-bilinear`. Chain `c15-joint-covariance-pilot-001`,
  children `seds-joint-covariance-smoke-001`→`seds-joint-covariance-001`.
- Six focused CPU tests passed; GPU parity/update smoke gates required before
  pilot. Background and yield after initial check, with logs and hard timeout.

Recent-paper queue note: [SignRep, ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wong_SignRep_Enhancing_Self-Supervised_Sign_Representations_ICCV_2025_paper.html)
metadata and abstract surfaced via official-CVF search. Sign-dictionary retrieval
is not sentence-levelPH retrieval. Check code/checkpoint/license/input cost before
adopting a pretrained encoder; not implemented, downloaded or claimed compatible.

Launched unix1789893705.5597959 (2026-09-20 15:41:45 local), timeoutPID3327357,
parent hard2660s. Cleanup completed2,198,712,420bytes/2.047711GiB permanently;
bothselectedbests retained, manifest `storage-prune-c15-refinement-20260920-001`.
V2 beforepilot~32.741GiB, plus3GiB+128MiB within36GiBcap; logs and source frozen.
Expected pilot~30min. Actual successful smoke/pilot charged only on collection.

Initial handoff check: smoke completed3updates/27.894715547561646s, zero-init
fullDEV score parity, both projection updates, nonzero input gradients and frozen
buffers passed. ParentPID3327358 alive, current child `seds-joint-covariance-001`
timeoutPID3328133. No polling loop; user will signal completion. Smoke runtime
remains reserved until collection, not yet charged separately.

Collected: completed666/allgatespass/noearlystop, selected444mean78.6127167630,
T77.6493256262/V79.5761078998; final666 same recalls. +.192678pporiginalC15,
tieGCN-R1seed42, -.096339ppbest1337. Defer currentpoolingfamily; no extra sweep.
Bothbest/last retained. Charge2143.77729249s once (smoke+pilot), parentexcluded.
