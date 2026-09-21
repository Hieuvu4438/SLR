# C16 — pretrained SignRep representation feasibility

## Material Passport

ARS experiment-agent/run,2026-09-20; public source/checkpoint installations
explicitly authorized. Goal remains actual SLRet gain, not asset-only completion.

Evidence: C13–C15 pose-local additions have not beaten GCN-R1. Try a richer
RGB sign representation learned with sign priors rather than further raw-geometry
or pooling variants. This is a hypothesis, not proof that frozen RGB is the cause.
Official source: https://github.com/ryanwongsa/SignRep (ICCV2025).
Releasev0.0.1 ckpt.pt222676746bytes, GitHub-provided SHA256
f8be8ca44aec4d7066175dccc681338f376ec86de6ba72bc79223a6bbd3c766b.
License retained CC-BY-NC-SA4.0; research use here, no commercial deployment or
relicensing. Source/weights stored under ignored artifacts; no new dependencies
installed into shared SEDS env. Existing timm/torch suffice for model imports.
Upstream example config strict-loads fullmodel; do not accept missing-head fallback.
Use weights_only=True and official checksum before loading downloaded checkpoint.

This tranche prepares exact native16frame-window features and latents (768D each),
frozenFP32 model, batch8, cacheFP16. RGB resize224/ImageNet normalization replicates
deterministic upstream evaluation; no nuisance/adversarial retraining or new labels.
Use same hash-fixed TRAIN512 as existing geometrypilot and allDEV519, noTEST.
Native retained-frame mappings and repeated padding preserved; cache per-video
window IDs and clip starts, validate frame counts/finite features. No RGB/2D asset
replacement and no generic extra-stream stack implemented in this tranche.

Next if feasible: choose a concrete transfer into native RGB representation and
matched original-feature control BEFORE training; avoid closed teacher anchoring,
confidence gating, local residual readouts and unmotivated stream stacking.
Current extraction alone cannot establish retrieval gain/novelty/SOTA. Official
paper's dictionary retrieval is not PHsentence retrieval; no direct scorecomparison.

Resources: existing other-user GPUjob untouched; allocator capped25%GPU (~12GiB),
batch8,4CPUthreads; checkpoint~.208GiB+source+<=256MiBcache fits36GiBstoragecap.
Parent3600s reserved, download1500s then smoke2videos cap180s. Extract1031videos
only if smoke wall/clip * fullclipcount *1.25 fits remaining parent. Otherwise
prepared_not_admitted; keep smoke/cache and report new allocation need explicitly.
Download resumes earlier partial16MiB; earliercurltimeout120s charged separately.
No polling, auto-retryofexperiments, fullTRAINexpansion, or retrievaltrain queue.

Tools: extract_signrep.py, prepare_signrep.py. Launcher:
`launch_bounded.py --name v2-c16-signrep-assets-001 --seconds 3600 --
/home/haipd/miniconda3/envs/seds/bin/python research/slret_goal_v2/tools/prepare_signrep.py`.
Parentartifact c16-signrep-assets-001; smoke signrep-native-window-smoke-001;
conditionalcache signrep-native-window-pilot-001. Charge parentwallonce onreturn.

Sourcecommit06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2 retained unchanged.
Model import succeeded in existingSEDSenv; inheritedplan validates1031videos/
60023nativewindows. Source-only check, NOT strictcheckpoint/modelGPU pass yet.
Launched unix1789896443.646114, timeoutPID3389889, hard3600s. Parent begins
resumable bounded download; no training process claimed before its gates pass.

## Pretrained ready / explicit extraction admission after user return

Assetparent terminal prepared_not_admitted,242.942352s. Download checksum exactly
matches release; fullmodel strict-load and GPU inference PASS105windows/2videos,
finite768D features/latents, peak2093412864bytes (~1.95GiB). No efficacy metric.
Original conservative estimate4203s included model startup repeatedly; log already
separates per-video costs1.802368s/41clips,2.464748s/64clips. Revised admission uses
60s startup + slower observed perclip rate *60023*1.25 =3358.278354s; no new
profiling run or relaxed data contract. Two samples/other-user load are limitations.
Remaining3697.266896s admits3500s extraction; no retrievaltraining budget assumed.
Launch unique `v2-c16-signrep-extract-001` with `extract_signrep.py --run-id
signrep-native-window-pilot-001`; exact same frozen extractor/source/pretrained.
On timeout retain complete caches, no implicit resume/retry. Return to user after
initial process check, not an active polling loop. User said pretrained download done.

## Registered C16 transfer/control contrast

Extraction completed1031videos (TRAIN512/DEV519),60023clips,1654.872425s;
171376889bytes NPZ. Entire cache shape/finite/windows/starts check passed.
This is asset feasibility only, no retrieval evidence yet.

Hypothesis: raw2D coordinate reconstruction and local pose pooling fail to teach
appearance-informed sign identity. Transfer public SignRep RGB features into
native GCN clip representations during training, so retrieval gradients can use
richer pose features without a third inference stream. Cause remains hypothetical;
2D may lack information needed to predict RGB and teacher may be domain-mismatched.

Native clip pose [B,64,1536] after existing sign_conv/mean gets an auxiliary
LayerNorm1536 + bias-free Linear1536->768 head (1182720 train-only parameters).
Target is detached frozen SignRep output 'features' for the EXACT16-frame window.
L = native seven-component total +0.1 * mean_valid(1-cos(prediction,target)).
No corruptions, text edits, inferred confidence, query weights, teacher rankings,
baseline score protection, gradient surgery or extra inference scorer.
Native RGB/pose2D/fusion/retrieval path preserved. Head discarded at inference.
Compare NO-GO protected-context: this predicts external input features, never
anchors baseline retrieval/logits/positives or protects a reference geometry.
Compare failedC08: external learned RGB targets at clip level, not missing rawXY.
Known cross-modal feature transfer is NOT claimed novel; a positive result would
be a stronger baseline needing further mechanism/ablation evidence.

One pair preregistered before scores: releaseinit,seed42,TRAIN512 hashsubset,
B32,tenepochs160updates, encoder1e-6/fusion1e-5/head1e-4, FP32moments.
Native GCN/dropout/BN eval-with-grad, no extra encoder unfreezing.
Control creates the same head under forked RNG but freezes/bypasses it and uses
the same data/order/native objective/LR/horizon. No DEVteacher targets used.
FullDEV selection only0/80/160; mean bidirectionalR1, eachdirection>=release-.5pp;
earliest ties, eligibleinit, stop ifmean drops>2pp. Existing best78.709056 remains.
TRAIN512 pilot cannot directly attribute comparisons to fullTRAIN7096 GCN-R1.
Read paired contrast/curves first; no family closure from undertrained singlepilot.
No SOTA/novelty/independent-confirmation claim from this exploratory DEV pair.

Small smoke proves auxiliary-only gradient, actualhead/GCN/fusion updates,
frozenbuffers, initial inference unchanged, and compactdelta load roundtrip.
All code/version provenance saved. Compact deltas require native release first;
include fusion,signbert.embed and transferhead, optimizer/RNG inlast.
Parent c16-signrep-transfer-pair-001 hard1800s: smoke120s then2x800s,
failclosed, no retries/parameter sweep. Authoritative reservation inSTATE/budget.
Keeplogs and yield after initialsmoke/alive; user returns when finished.

Actual execution: parent001failed21.116541s before anytraining. Smoke17.821749s
failedhistorical step0score gate: RGBexact, maxpose.1628227234/fusion.0973358154;
pose differences>1e-4 in7video rows, causeunresolved. Preservefailedcode/logs.
Do not relax tolerance or infer SignRep efficacy failure. Bothtrainingarms unstarted.
Scope next repair to newhook/input/runtime numerical path, not another baselineaudit.

### Scoped evidence and pair002 contract amendment (before training)

check_signrep_hook002 compares first32DEVitems (includes historically differingrow1)
with one model/batch: native repeat, before/after headallocation, and hookenabled/
disabled are all bitwise identical for masks and both encoded visualstreams.
This proves the inspected changed path, NOT cross-process equality on allvideos.
Diagnostic001 failed because TRAINunpacker expected augmentedtext onDEV;002 uses
native visualinput dictionaries. Both runtime costs and errors recorded inbudget.
Failedsmoke initialR1/5/10 match historical values for ALLstreams/directions;
fusionMeanR also matches, poseV2TMeanR differs1/519. Score difference cause unresolved.

V2§4/§7 does not require historical score replay for an unchanged inference path.
Amend validation before training: require exact scoped hookproof; freshly evaluate
fullDEV step0 in BOTHarms; require allhistoricalR1/5/10 matched; save freshmetrics/
scores as each arm's actualinit; record crossprocess score differences as limitation.
This replaces the overbroad historical score-equality gate explicitly, not a fix
to the unexplained numeric variation and not a raised tolerance. Native references,
trainingloss/subset/LR/horizon/selector remainunchanged. Pairedinitrecalls mustmatch.
No inference improvement or novelty claim. Preserve failedpair001 anddiagnostics.
Pair002 admitted1800s: smoke002 then originalplannedtransfer001/control001;
bothlongarms were absent atlaunch. Stop on gradient/update/cache/recall failure.

### User-requested OOM repair/retry

Pair002 smoke passed; transfer001 failedCUDAOOM duringforward23 after22updates,
beforefirstposttrainDEV. Controlunstarted. Not a negative method result.
Use torch2.3.1 installed save_on_cpu(pin_memory=True) around forward to store
autograd savedtensors on hostRAM and retrieve them onbackward. No microbatching,
changednegatives, reducedframes, changedprecision/loss/LR/seed/horizon.
Do not claim numerical equivalence across differentprocesses; prior caveat remains.
InitialDEV stillfresh and recallgated. Smoke uses failedrun batch_indices[22]
twice, checks existinggradient/head/GCN/fusion/buffer gates and <=16GiBpeak.
Record CUDAallocated/reserved/peak andhostRSS eachstep. Periodic atomiccompact
last.pt every16steps includesoptimizer/RNG/batchindex; no resume capability claimed
until a resume is actually implemented/verified. This run restarts releaseinit
because failedrun had no savedtrainingcheckpoint.
Boundedparent c16-signrep-offload-retry-001 smoke150s thenpilot1500s, total1700s.
Onlytransfer pilot now; matchedcontrol must use sameoffload recipe later, ifbudget.
Preserve allfailedruns and cache/bests. User explicitly says no polling: check
startup once, hand off logs, then wait for user to reportcompletion.

Offloadretry001 outcome: timeout124 at150s; fullDEV took133.293906s, no completed
trainingstep and no loggedOOM. Parentterminal151.531626s, childrun.json remained
stalerunning afterSIGTERM; worker absent. This is an inadequate smoke timebudget,
not evidenceoffload failed. Keep rawreports/logs unchanged forprovenance.
Retry002 memorysmoke skips redundantfullDEV and tests only failedTRAINbatch23
forward/backward twice. Existinggradient/update/buffer/16GiBpeak gates unchanged.
No DEVmetrics claimed fromsmoke. MainpilotstillfullDEV0/80/160 withactualinit.
Parent1600s (smoke150s,pilot1400s,50s overhead), within1650.561713s remaining.
No new topup/control job/TEST. Checkstartup once and no polling thereafter.

Retry002 reached backward but failed efficientattention maskstride alignment:
attn_bias.stride(1)=4225 (65x65), kernel requiresmultiple4. CPU savedtensorpacking
can densify expandedviews. PeakCUDA2953653760bytes (~2.75GiB), notOOM.
Parent26.178253s charged. Retry003 selects mathSDPA only insideoffloadedtraining
forward (flash/memoryefficient/cudnnSDPA disabled); inference remainsnative.
Same attention formula, batch/loss/LR/horizon; kernel numerics/runtime may differ,
so matchedcontrol needs sameoffload+mathSDPA. No claim of bitwise trainingparity.
Preserve failedlogs; validate actual failedbatch forward/backward beforepilot.

## Collected result / next decision (GoalV4)

Offloadretry003completed: smoke2steps46.776685s, transfer160steps1199.781731s;
parent1258.412504s chargedonce. No OOM/earlystop; gradient/head/GCN/fusion/buffer
gates passed. Peak6.387GiBGPU/27.348GiBhost. Selected160meanR1=78.0346820809;
T77.071291/V78.998073. Curve0/80/160=77.552987/77.842004/78.034682.
Gain+.481696pprelease, still-.674374ppbestGCN-R1. Best+last retained andbesthash
verified6f2fda6dfba74b5ffdb67d15f5c3c670545707a0765a298dac207fd7380dbcc1.
Hypothesis remains unresolved: gain may be ordinary GCN/fusion continuation, not
externalrepresentation supervision. Auxiliaryloss falls .978503->.546527 (first/
last10means), but that does not establish useful transferred knowledge.
Next decisiveexperiment: sameTRAIN512/B32/160steps/seed42/nativeinit/GCN-fusionLR/
offload+mathSDPA/DEVselector withtransferheadfrozen and noauxloss. If similar/better,
do not credit SignRep; if transferwins meaningfully, prioritize fullerdata validation
and replication before expensive novelty claims. NoLR sweep now. Only365.970956s
remain vs~1200s measuredpilot, so WAITING_FOR_USER for bounded30minute allocation.
