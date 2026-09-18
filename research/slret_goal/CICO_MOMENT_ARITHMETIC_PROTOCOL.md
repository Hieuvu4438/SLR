# CiCo real-gradient moment arithmetic activation

2026-09-18 preregistration, ARS experiment-agent inline. This is numerical
baseline/empirical-mechanism feasibility, NOT a new optimization method.
Motivation: optimizer-exposure-001 verified FP16 stored-state phenotype in
CiCo PH/CSL. Mere zeros do not prove underflow or retrieval harm.

Use the already audited selected CiCo PH checkpoint first; CSL uses its own
selected checkpoint if the PH runtime succeeds. Hash-lock selected checkpoint,
config, TRAIN manifest and all features consumed. One seeded32-example TRAIN
batch per dataset, native feature/caption/text-augmentation/amp_bf16 paths,
existing balanced CLCL objective and outerclip1. CSL samples one video per
captiongroup. No auxiliary/teacher loss, no DEV/TEST, no model updates. This
reduced batch is a mechanism screen, NOT the original B512 training exposure.

Capture one real gradient and reuse EXACTLY that tensor for all arithmetic
arms; no distinct forward/RNG trajectories. Save clippedgradient tensor/hash
and batch IDs to permit independent CPU/GPU arithmetic replay. Keep every
source parameter unchanged. Use each checkpoint's stored optimizer group LR,
betas/epsilon/weightdecay, but explicitly initialize fresh zero moments and
step1 for this diagnostic; do not pretend to repair historical optimizer state.

Compare tensor-by-tensor GPU AdamW using installed PyTorch functional kernel:
1. FP16parameter/native-dtype gradient and moments, foreach=False.
2. Sameparameter and identical represented gradient promoted to FP32;
   moments FP32, foreach=False (no FP32 master parameter).
3. AllFP32 parameter/gradient/moments reference.
Also compare native foreach=True on the same tensor to expose execution-path
rounding. Never attribute its difference silently to moment precision.
FP32 nativeparameter tensors serve as a zero-difference control.

Record m!=0/v==0 exposure; specifically count nativev==0/referencev>0,
showing loss by arithmetic/storage in this step rather than unvisited coordinates.
Report normalized-update error (before parameter rounding) and actual weight
delta differences; separately fix first moment to native values and swap ONLY
second moment denominator to isolate that arithmetic contribution.

Activation gate: finite actualgradients, originalmodel unchanged; native vs
FP32moment counterfactual differs in relativeL2 actualweightdelta by>1%, and
>1% of FP16elements have nativev==0/referencev>0. Must show fullFP32negative
control and nativeforeach discrepancy separately. Passing only admits a matched
training feasibility plan; it cannot establish recall gain, novelmechanism or
paper contribution. Ifgradientzero/no exposure, do not rescue bychoosinglayers.

Budget per dataset <=300s,<=384MiB savedgradient/report,CPU/GPU local only.
Currentcampaign22.435GiB +2*0.375GiB<24GiB, freefilesystem145GiB, reserve15GiB.
Atmost2GPUdiagnostics; charge actualtime todiscovery8624.686s remaining.
No automatic retry onunknown failure, no longtraining admission. Source/tool
errors may be repaired with failedrun retained under user research authorization.

## Native-batch fidelity check registered after B32 results

B32 PHpasses, CSLfails exposure(0.099354%<1%). These results stay unchanged.
Both selected checkpoints have nearzero B32loss, whereas their actual training
usesB512/511negatives. Historical run_summary.json on bothdatasets reports
43,171,500,032bytes peakGPU; local49,140MiB device is idle and fits that bound.
Before any causaltraining admission, inspect EXACTLY one original-sizeB512
batch per dataset, taking the first512 of the SAME registered permutation/
grouped sampler (B32 is the prefix). This tests an identified exposure mismatch,
not a batch-size search; no sizes beyond32andnative512, no layer/seed selection.
Keep all arithmetic arms and gates unchanged. Report B32failure alongsideB512.
Passing cannot retroactively validate the reduced-batch claim.

Eachfullbatch<=300s and<=384MiB (gradient shape independent ofbatchsize).
With allfour gradient artifacts,22.435GiB+4*0.375=23.935GiB<24GiB. CPUstartup
failure0.702426s remains inledger; no otherfailedrun retried automatically.
No model updates or DEV/TEST, no training sweep admitted by this amendment.
