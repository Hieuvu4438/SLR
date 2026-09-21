# C14 — explicit bone descriptors in the native pose encoder

- Material Passport: ARS experiment-agent/run, registered2026-09-20; user V2 model-improvement authority. No TEST or new corpus extraction.
- Hypothesis: absolute XY GCN may not expose local finger/arm geometry effectively. Both C13 topology settings improved standalonepose but did not beat GCN-R1 fusion; stop graph-LR refinement and test explicit geometry instead. This does not prove an information deficit.
- Prior: [Shi et al., 2s-AGCN, CVPR2019 §4.4](https://arxiv.org/html/1805.07694v3) defines parent→child bone vectors. Relevant section read, not fullpaper reread. Borrow representation only; no fulltwo-streamAGCN reproduction/novelty claim.
- Exact descriptors per joint: `(dx/256,dy/256,length/256,dx/max(length,1px),dy/max(length,1px))`, from native directed anatomical parents; rootselfedge zero. Zero-padded endpoint pairs excluded, not a learnedconfidence gate. Nonfiniteinput fails closed.
- Operator: bias-free zero-init5→128 projection added to first graph-convolution output, before nativeTCN/BN/ReLU; shared for bothhands, separatebody. Downstream pretrainedGCN/pooling/fusion unchanged; preserve original absoluteXY/RGB. Adds1280parameters, no graphdelta.
- Difference from C13: explicit nonlinear length/direction feature map, not learnedadjacency. Difference from C07: no QKV or crossarticulatorexchange. Difference from C09: native2DfullTRAIN, no3Dcachedstream/postencodedresidual. No gallerygraph, confidencegate, C06temporal or nuisance-removal objective.
- Base release, seed42/B32/TRAIN7096/DEV519,3epochs666steps. GCN1e-6/fusion1e-5/newprojection1e-5, native losses/FP32moments, fixedGCNBN/dropout. No extra augmentation/supervision or frozenposecache.
- DEV0/111/222/444/666, primary mean(T2VR1,V2TR1), initincluded; eachdirection>=release-.5pp; earlymean drop>2pp. Compare selected and samefixedsteps to historicalR1seed42 plus globalbest78.709056. R2 replay mismatch remains causal-comparison limitation.
- Gates: CPUdescriptor/tree/padding/tinybone/nativeparity/gradient/serialization/LR tests; actualtwo-stepGPU must update both projections/nativeGCN/fusion, fixedbuffers; fullDEVzero-init once in freshsmoke then identicalsource/base/data/configproof inherited.
- Cost: expected~30min, estimatedpeak18GiB, smoke90s/pilot2300s parent2430s hardcaps. Remaining4298.220139s after C13-R1; reserve2430s, unreserved1868.220139s, no auto top-up/cloud.
- Disk:3GiBpilot+128MiBsmoke; retire only unusedC13-R1last and originalC04last, retain selectedbest of both, allGCNseeds/data/logs. Cap36GiB/floor15GiB unchanged; no newassetdownload needed.
- Driver: `python research/slret_goal_v2/tools/run_adaptive_graph.py --bone-features` (reuse bounded launcher); chain `c14-bone-features-pilot-001`; children `seds-bone-features-smoke-001` and `seds-bone-features-001`.
- Promoteonlyprovisional if fusedgain exceeds matchedR1; originalXY/noadapterR1 is absentcomponentcontrol. Before geometry-specific claim, need equalcapacityrawXYprojection control and replication, budget permitting. Feature-only posegain is insufficient.
- Risks: normalizeddirection noisy for tinybones, first-layer injection may disrupt pretrainedscale, initialLR may underlearn; roots retained through unchangedXY. If no usefulfusedsignal, defer exactspecification, no blind strength/gridsearch.
- Launchbackground, confirm initialsmoke/alive once then yield to user. No automaticfollowup jobs while waiting. Novelty/generalization/SOTA unresolved.

## Startup

12 scopedCPUtests passed. GPU smoke2steps22.8513324261s completed withfullDEV
zero-init parity, bothboneprojections/nativeGCN/fusion updates and fixedbuffers
passed. IntendedboneLR1e-5/1280params verified. Pilot observedalive step5,
trainerPID3154569, inheritedparity/updategatespass; no efficacyresult yet.
Charge smoke+pilot once at collection; neither charged in31593.779861s yet.
Cleanup removed2043322862bytes/1.902993GiB permanently, retainedbothselectedbests.
Log `artifacts/slret_goal_v2/c14-bone-features-pilot-001/seds-bone-features-001.log`.

## Collected result

Completed666/1713.533791s, allgatespass/noearlystop. Selected444mean78.612717,
T77.842004/V79.383430, exacttieR1seed42; globalbeststill78.709056. Final78.516378.
No newincumbent. Bothprojectionsupdated; norms at660hand.01046447/body.00709206.
One strongerprojectionLR contrast registered in METHOD_BONE_FEATURES_C14_R1.md,
not provenundertraining; no blind sweep. Children1736.385123s charged once,
remaining2561.835016s. C14bestretained, unusedlast retired for nextpilot.
