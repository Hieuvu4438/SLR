# Same-I3D retrieval fine-tuning: two-step smoke preregistration

2026-09-18. Material Passport: ARS experiment-agent, inline run; activation only.
Previous turn PROGRESS: fixed-eight-window gradient replay verified, not efficacy.

Keep SEDS release, native loss/augmentation, firsttwo shuffled B32 batches of
moment-control001, seed42, native head optimizer with FP32 moments unchanged.
Same I3D BSL5K/Mixed5b+5c, full recorded frames/windows, no new stream/scale,
teacher, labels, objective or scorer. B_tuned feasibility, not novel method.
Collision scope and attribution remain RGB_GRADIENT_FEASIBILITY.md; generic
fine-tuning/GradCache are existing techniques. No candidate gate consumed.

SEDS head stays in native torch2.3.1; persistent RGB subprocess uses original
torch2.11 so initialization features are bitexact. A versioned local-file pipe
passes float32 features and gradients. Each step encodes all32 videos, computes
one native B32 loss, replays allwindows with fixedmicrobatch8, and only then
updates both optimizers. No stale cachedfeature reuse after a weight update.
Forward frames may remain in CPU RAM for same-step replay; not persisted.

I3D tail optimizer is vendored BertAdam, float32 params/moments, LR1e-5,
b1=.9/b2=.98/epsilon1e-6, cosine warmup.1/222steps, weightdecay.001 except
bias/BNaffine (zero); tailglobalclip1 and nativeperparameterclip1. Head clipping
unchanged and separate from tail. Firstwarmup step LR0 must leave both weight
sets unchanged; secondstep must change both. This fixes one activation recipe,
not an accuracy hyperparameter search. BatchNorm/dropout remaineval in I3D.

Gates: initialization features maxabs<=1e-4 against recordedfeatures; replay
features bitexact to own forward; firsttwo losses within1e-7 of matched control
(both see initial I3D because step1LR0); finite gradients; nonzero tailupdate at
step2; frozenweights/BNbuffers unchanged; version/order assertions; regenerated
firstvideo of secondbatch changes after step2. Report every loss/exposure and
RNG/endstate provenance. No DEV/TEST/evaluation/selection in this smoke.

Version protocol: encode(versionN) -> gradient-ready(versionN) -> commit(N+1).
Reject stale/duplicate/out-of-order requests. Head and RGB commits are NOT
crash-atomic across processes; any interrupted/partial commit invalidates run
and cannot resume. Smoke retains tail+optimizer but not head checkpoint, hence
is explicitly nonresumable. Failures preserved, no automatic hidden retry.

<=600s wall, <=128MiB output (features/gradients two batches~32MiB, tail+moments
~55MiB, logs). Campaign remains15GiB with15GiB free-space reserve. No deletion
or cap expansion. Record bothprocess peaks and full stepwall/RAM; summed CUDA
allocation is conservative, not simultaneous measured peak. CPU tests before
launch. A full222step pilot needs separate registration and retention budget;
two-step throughput does not establish recall efficacy.
