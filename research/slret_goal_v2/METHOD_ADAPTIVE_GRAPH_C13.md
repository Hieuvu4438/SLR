# C13 — adaptive intra-articulator joint topology (registered 2026-09-20)

- Hypothesis: fixed native joint adjacency limits task-specific finger/body relations; positive three-seed GCN adaptation motivates adapting topology too.
- Operator: each of three pre-pooling graph convolutions in shared-hand/body encoders uses `A+B`; B is trainable, zero initialized, same shape as native A. Physical/pooled graph buffers unchanged.
- Prior: borrow only global B from [Shi et al., 2s-AGCN, CVPR2019, §4.1 Eq3](https://arxiv.org/html/1805.07694v3). Relevant section read; not a full-paper-read claim. No sample-specific C, bone stream or full reproduction; novelty unresolved.
- Site differs from C07 post-GCN cross-articulator attention: within each graph convolution, no QKV, joint exchange, teacher or scorer residual. No collision with gallery-graph C43 or user-closed C06 temporal delta; native GCN also adapts here.
- Base: native `third_party/SEDS/ckpt/ph_best_model.bin`; preserve RGB+all pose2D. Corrected H4W3D assets remain retained, not used in this contrast.
- Recipe: release init, TRAIN7096, seed42, B32, 3epochs/666 native updates, native losses/FP32 optimizer moments; GCN1e-6/fusion1e-5, new B1e-4. GCN BN/dropout fixed eval. LR is an explicit initial hypothesis for zero-init new parameters, not tuned evidence.
- DEV519 full-gallery at0/111/222/444/666; init included; select mean(T2V R1,V2T R1), each direction>=release-.5pp. Early stop if mean drops>2pp below release; no TEST loading/tuning.
- Compare selected and common fixed steps with historical R1 seed42 78.6127167630, plus global incumbent78.7090558767 and release77.5529865125. Same base/data/order/schedule checked. R2 exposed unresolved replay drift: historical control is exploratory, not exact causal proof.
- CPU native-operator identity/gradient/state-reload/LR tests; two-step realGPU smoke verifies all six B tensors update, GCN/fusion update, frozen buffers fixed. FullDEV zero-init check once in smoke; source/base/data/config-matched pilot inherits proof.
- Cost: smoke90s cap + pilot2300s cap; parent2430s reservation, expected pilot~29–35min, estimatedVRAM<12GiB; smoke128MiB + pilot4GiB conservative disk reservation, cap36GiB/floor15GiB retained.
- Continue/replicate if increment over R1 has promising curve; no automatic promotion from release-only gain. Otherwise inspect update magnitude/curve for at most one motivated next refinement before changing family; no blind LR sweep.
- Risks: global adjacency may overfit signer/data, extra edges may hurt pretrained graph features, new LR may be too large/small; improvement and generalization remain unmeasured.
- Driver: `python research/slret_goal_v2/tools/run_adaptive_graph.py`; chain `artifacts/slret_goal_v2/c13-adaptive-graph-pilot-001`, children `seds-adaptive-graph-smoke-001`, `seds-adaptive-graph-001`. Background, persistent logs; yield after initial smoke/alive verification, collect on user return.

## Additional authority

User2026-09-20 permits installing external public repositories/dependencies and downloading checkpoints/pretrained models as needed. Check source/license/compatibility and disk before use; does not authorize cloud spend or unlimited GPU. No external assets needed for C13.

## Startup outcome (not efficacy)

Nine CPU tests pass. GPU smoke completed2updates/27.9632246494s: fullDEV
zero-init parity, all six graph matrices updated, intended GCN/fusion updates,
frozen buffers and LR groups passed. Actual new parameters5880 (nativeK=4);
peak allocated14638843392bytes (~13.63GiB), above initial12GiB estimate but
within48GiB GPU. Pilot observed running step3, PID2868519; hard childcap2300s.
Smoke runtime not yet charged; collect both children once on user return.
No training efficacy conclusion yet. Persistent log:
`artifacts/slret_goal_v2/c13-adaptive-graph-pilot-001/seds-adaptive-graph-001.log`.

## Collected result

Completed666/1808.982913s, noearlystop; selected444mean77.745665, final77.552987.
Selected+.192678pprelease but-.867052ppR1seed42/-.963391ppglobalbest. Posefinal
64.643545 vsrelease60.500963; no fused gain over R1. All implementation gates
passed; peak17.306579GiB. OriginalC13best/logs retained, no promotion.
Children1836.946137s charged once; remaining6086.978399s. One slowergraphLR
contrast registered in METHOD_ADAPTIVE_GRAPH_C13_R1.md; causal explanation
unproven and no blind sweep. This original recipe is complete, not still running.
