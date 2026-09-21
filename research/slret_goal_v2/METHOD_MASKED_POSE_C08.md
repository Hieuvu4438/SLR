# C08 — train-only masked pose reconstruction

## Completed pair002: defer masking, retain clean GCN lead

Treatment222steps618.296536s selects111mean77.745665; clean control222steps
591.521164s selects222mean78.034682. Same order/config, all gates passed.
Treatment/control at111 tie77.745665; at222 treatment77.552987 vs78.034682.
Reconstruction loss decreases, but finalV2T falls below guardrail. Thus combined
masking+reconstruction under this recipe did not improve the matched baseline.
Defer current recipe, preserve provenance. Control is new provisionalDEVbest,
+.481696pp release and+.096339pp C04, NOT a masked-objective gain or SOTA.
Next C11 tests composition of clean GCN tuning with existing C04 visual LoRA.
All numbers/limitations/child cost accounting in RESULTS.md and current STATE.md.

## Attempt001 implementation failure / authorized repair002

Material Passport: academic-research-suite/experiment-agent/run, inline,
2026-09-19, user explicitly requested fix and rerun. Attempt001 failed at first
backward before any optimizer update, child20.801845s. This is not model evidence.
The auxiliary decoder loss was stored off the returned forward graph, conflicting
with native DDP find_unused_parameters. Repair returns it as an eighth output
alongside the unchanged seven native losses. Runner performs one backward only;
remove separate autograd.grad probe, check auxiliary-only feature-edge gradient
via hook plus native encoder gradients/updates. Plain CPU auxiliary-only test
also confirms encoder gradients; actual Gloo/DDP two-step regression covers both
arms. Five focused tests passed0.468s. No static-graph bypass or disabled DDP.
Newly attached module inherits model eval mode, preventing accidental masking
if attached to an already-evaluating model. Existing inference remains unchanged.
Rerun: tools/run_masked_pair.py --attempt 002, new chain/child suffix002, old
artifacts preserved. Same seed/data/mask/loss/LRs/horizon/limits. Failed child
charged once; remaining4601.879926s,currentpool2527.263531s,reserve2460s.
GPU smoke002 completed2updates27.841263s, nativeDEV score parity and all
gradient/update/buffer gates passed. Chain then started treatment002; retrieval
improvement still unmeasured. Background/log/yield; no training monitoring loop.

## Material Passport / status

academic-research-suite, experiment-agent/run, inline execution, 2026-09-19.
Exploratory local adaptation; no efficacy, novelty or SOTA claim. Four focused
CPU tests pass0.418s; real-data gradient/zero-inference smoke gates pilot.
Authority: user V2 goal and permission to borrow/combine existing paper ideas.

## WHY / prior art / limitation

C07 pre-pool attention and C09 additive geometry did not beat C04. Instead of
another added branch, test whether explicit spatial recovery supervision makes
existing GCN representations more useful with incomplete pose observations.
This is a hypothesis, not a diagnosed cause of baseline errors.

Borrowed mechanism: [Scaling up Multimodal Pre-training for Sign Language
Understanding](https://arxiv.org/html/2408.08544v1), encoder/decoder and loss
description around Eq3: masked-joint reconstruction with a two-layer decoder,
jointly with sign-text contrastive learning. Original primary HTML inspected
via browser on2026-09-19, not a systematic review or human-read certificate.
Its data scale, confidence-weighting and architecture differ from this pilot;
reported gains do not transfer to our protocol. Reconstruction itself is prior
art, not our novelty. If effective, attribution/generalization precede contribution.

## HOW / implementation contract

Native loader supplies 49 XY joints (left21/right21/body7) in256px crop units.
Confidence is discarded upstream. Exclude nonfinite coordinates (fail), all-zero
XY joints, and frames outside the union of native valid16-frame clip windows.
Do not fabricate confidence or use held-out outcomes to choose targets.
Mask independent20% valid joints to zero only during training. Dedicated seeded
mask RNG preserves native data/dropout RNG sequence between treatment/control.
Retain native GCN, temporal processing, RGB path and scorer. Decoder attached
after GCN per-frame1536d pooled features: LayerNorm -> Linear128 -> GELU ->
Linear98, predicting XY/256-.5. MSE averaged over masked joints/coordinates only.
Loss = unchanged native retrieval objective + .05 * reconstruction. Single
predeclared coefficient; target normalization makes its scale explicit. Log
raw and weighted loss, masked fraction, finite gradients and encoder updates.
This initial small mask ratio is a conservative fine-tuning choice, not the
paper's optimized setting or a verified optimum. Coordinate bias/noisy targets
and corruption of retrieval features are known failure risks.

Train native signbert.embed weights1e-6 and fusion1e-5; decoder1e-4 treatment
only. GCN stays eval (BN/dropout fixed) with requires_grad true; upper visual,
text/RGB encoders and sign-conv remain frozen. Explicit optimizer name partition
avoids native sign_lr accidentally setting fusion LR. Decoder bypassed in eval;
inference structure remains native, no new features required. Checkpoints store
full state; masked_pose keys are train-only and may be omitted for deployment.

## Decisive comparison / protocol

Release initialization, full TRAIN7096/native augmentation, B32 seed42,
1epoch222 updates. Control has identical native trainable weights/LRs/order,
but no masking/reconstruction; inactive decoder initialized identically under
forked RNG. Contrast establishes combined masking+objective effect, not the
reconstruction-specific contribution. If promising, require mask-only control
before attributing gains to auxiliary supervision. Do not combine C04 yet.
Full DEV519 at111/222, initial reference reused after smoke checks full-gallery
score parity <=1e-4. Primary mean bidirectionalR1, report R1/5/10 bothdirections.
Reference77.5529865, C04incumbent77.938343. Selection includes initialization;
each direction >=reference-.5pp, earlystop mean deterioration>2pp.
No TEST loaded or analyzed. Repeated DEV selection, one seed and short horizon
limit claims. Compare selected AND common fixed evaluations; if either arm
early-stops, do not represent unequal realized horizon as fully matched.

## Execution / resources / next decision

Entry tools/run_masked_pair.py -> train_seds_extended.py --masked-pose
{reconstruct,control}. Chain c08-masked-pair-001, children seds-masked-smoke-001,
seds-masked-pose-001, seds-masked-control-001. Smoke2steps checks finite loss,
auxiliary gradient reaching GCN, expected parameter updates, frozen buffers and
unmodified inference. Failed implementation gate stops chain, no automatic retry.
Limits240/1080/1080s; parent2460s <= currentpool2548.065376s; remainingtotal
4622.681771s before this run. Estimate1100–1800s, GPU<=48GiB, newstorage<=8GiB,
disk76GiBfree with15GiB floor, V2totalcap36GiB. Preserve C04/release and C09
cache/checkpoints; no deletion needed. Charge child elapsed time once on return.
Launch background, persist logs, check alive once, yield as requested. No
automatic next experiment or monitoring loop. On return: compare reference,
control,C04; defer if no lead and no motivated repair, otherwise targeted control
or <=3 justified refinements, not a blind hyperparameter sweep.
