# AS-C28 — clean calibration versus local upstream training semantics

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / code audit
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C28-v1

Audit the local upstream default2D/freeze_layer_num0 path, not an asserted exact
PH release training recipe. README default training example is How2Sign. AS-C20
was a distinct bounded calibration, not upstream replication; preserve that label.
No actual training campaign, dev/test input or PH retrieval checkpoint required.

1. Load exact AS-C19 generic initialization. Execute ONLY the parsed upstream
   freeze-policy AST block with args(freeze_layer_num0,linear_patch2d); inspect
   actual named parameters before/after. Require frozen keys precisely text
   token_embedding.weight and positional_embedding, report shapes/counts.
2. Compare those tables to AS-C20 final TRAIN-only checkpoint. Report relative
   L2 and absolute drift, without claiming harmful forgetting. Weight decay can
   move unused rows; nonzero drift is not evidence of semantic gradients.
3. Classify upstream default coefficient1 weight-decay groups using its exact
   name predicate (`bias`, `LayerNorm.bias`, `LayerNorm.weight`) versus AS-C20
   exclusion of biases and all1Dparameters. Report keys/counts, no retraining.
4. Execute actual local BertAdam and installed torch AdamW on identical fixed
   float64 CPU parameters/gradients, beta(.9,.98),epsilon1e-6,lr1e-4. First use
   constant lr, zero weight decay and disabled clipping to isolate moment/bias
   semantics; compare BertAdam with an independent explicit uncorrected formula.
   Next use the registered1000-step schedules/weight decay=.001 with global
   clipping in BOTH caller paths, upstream's extra internal clipping, and record
   updates1/2/10/100/500/1000. This second comparison has multiple known factors;
   do not attribute its total difference to bias correction alone.
5. Record first/peak/final schedule factors, name-based learning rates, and the
   upstream training-loop logit-scale clamp versus AS-C20's observed final scale.

Upstream globally clips before optimizer.step; internal per-parameter clipping
is not a substitute for that caller behavior. AS-C20's new random tensors have
lr1e-4, generic tensors1e-5; upstream coef_lr1 gives all clip tensors lr1e-5 by
default. Upstream warmup uses per-parameter prior step index starting0 and cosine
of total progress, AS-C20 uses update1..1000 and reparameterized post-warmup cosine.

Pure synthetic update/AST/shape tests only. Mathematical discrepancies are
implementation evidence, NOT evidence any difference caused poor retrieval or
that matching defaults will improve it. Any subsequent training requires a
separate controlled protocol; freezing/optimizer corrections are not novelty.

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.training_recipe_audit`
Cwd `/home/haipd/SLR`; timeout300s; monitor process plus`AS-C28-RECIPE_run.json`.
Save JSON only; no model changes persisted. Source hashes frozen in report.
Preserve any failed assertion/attempt; no silent retry.
