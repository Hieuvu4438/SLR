# SEDS adapted baseline continuation — preregistration

2026-09-17. This is a strong control, NOT a method candidate, not B_release,
and not a reopening of augmentation/fusion/uncertainty mechanisms. Goal: measure
whether ordinary continuation on matched adapted TRAIN/DEV outperforms the
released checkpoint transferred to adapted inputs. Initial mean devR1=77.552987.

Use unchanged native model/loss, released PH initialization, random_swap text
augmentation, fusion/pose/RGB losses, rgb_pose_match=.4, KL disabled. Reuse
native BertAdam groups and decay, CLIP LR1e-5, SignBERT/default LR1e-4,
warmup cosine .1, grad clipping1, clamp CLIP logit_scale to log100. Seed42.
One GPU batch32, no gradient accumulation (32 in-batch negatives, explicitly
different from distributed release128); max one complete shuffled TRAIN pass,
222 updates including last batch24. No AMP beyond native parameter dtypes.

2026-09-18 source audit clarification: native forward weights both auxiliary
text-stream losses by1, despite parser alpha=.8 and paper Eq4 describing.8.
This control intentionally retains the actual native forward. No silent
paper-faithful loss correction; that would be a separately labeled control,
not a novel method. Retrieval consequences of the mismatch are UNKNOWN.
Shuffle and augmentation RNG state saved; no data-worker nondeterminism.

Smoke first32 canonical TRAIN examples, two repeated optimizer steps, no
retrieval-gain claim. Record loss, gradients, tensor updates, VRAM/time; preserve
checkpoint with optimizer/RNG. Full continuation starts independently at release
initialization, never from smoke state. Smoke must validate a true nonzero
optimizer update (warmup may make first scheduled LR zero).

Full run requires complete verified7096 adapted TRAIN and519 DEV, identical
extractor/model/preprocessing contracts. Evaluate step0, step111, step222.
Step0 score parity maxabs1e-4 versus existing dev-eval002; all metric differences
<=1e-5. Select highest dev mean bidirectionalR1, ties favor earlier step including
initialization; guardrail no direction below step0 by>0.5pp. Report allR1/5/10
and both trained checkpoints regardless of selector. No test access. Training
gain only if selected step>0; +0.5pp pilot lead is descriptive at this exposed
dev, not confirmation. No hyperparameter rescue if baseline does not improve.

Budget: smoke<=5min, full continuation<=20min, counted within4h discovery/matched
controls (not the2h feature-parity tranche). No new method configurations spent.
Store best/last and evaluation matrices; at most two trained full state files,
no step0 duplicate. Estimate disk before launch (15GiB free reserve and15GiB
campaign artifact cap). Exact tensor hashes, per-batch IDs and checkpoint/RNG
state enable audit; full resume requires matching config/source/assets and next
batch index, and must never restart finished steps silently.

This released model has already seen original TRAIN; internal holdouts carved
from it are not fresh confirmation. Future confirmation must use final locked
test or training from initialization with disjoint groups. No SOTA comparison
between this dev control and published test numbers.
