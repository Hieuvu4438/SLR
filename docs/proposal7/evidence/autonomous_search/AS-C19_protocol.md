# AS-C19 — PH-unfitted initialization and internal holdout feasibility

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / provenance audit
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C19-v1

The earlier R0 is PH-fitted and provides almost no distinguishable train errors.
This audit tests whether a different diagnostic can avoid that fitted checkpoint;
it is NOT the rejected AS-C07 cohort-teacher candidate and supplies no teacher
targets. No model improvement/GO or global pretraining-contamination claim.

## Fixed checks

1. Hash local generic `ViT-B-32.pt`, expected40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af.
   Inspect TRAIN-only I3D extraction reports and match checkpoint file hashes.
   No PH/H2S/CSL retrieval-release checkpoint is loaded by this audit.
2. Instantiate baseline Filip architecture using generic CLIP dimensions and
   shared task config. Copy only exact-shape generic CLIP tensor keys. Explicit
   expected random keys: visual.conv1.weight, visual.positional_embedding,
   visual.conv2_trans.weight. Reject any other missing key or non-metadata
   unexpected generic tensor. No silent shape-mismatch acceptance. All copied
   tensors must equal their generic source cast to target dtype exactly.
3. Seeds42,42,1337: repeated seed must reproduce random-key hashes; alternate
   seed changes all three random tensors, copied weights invariant. No checkpoint
   saved; metadata only. This custom explicit mapping is not claimed to be the
   official release's full training initialization recipe.
4. PH TRAIN only: partition source prefixes by SHA256 mod5, fold0 held out.
   Report every index/ID and fold counts, source overlap0, exact deployed-text
   overlaps. Also count source+exact-text connected components for diagnostic
   feasibility only; do not select a partition after seeing counts. Prefix is
   inferred recording identity, not independently verified recording metadata.
   No official split or positive is changed; this is an internal train audit.
5. Load first32 fit-partition rows from existing fused I3D train features and
   run generic-initialized core in explicit float32 eval mode. Clean and deployed
   augmentation seed42/epoch0, historical four-CE objective. Finite encodings,
   both losses, gradients; no optimizer step. Report actual memory/time and
   target coverage, not anticipated full-training performance.

Hash provenance does not prove generic CLIP/I3D pretraining excludes every PH
clip, speaker or web derivative. The supported claim is absence of a deliberately
PH-fitted retrieval checkpoint in this loader. Long-training learning curve,
model-selection protocol, actual heldout residual availability and probe transfer
must be measured later; near-random initial errors would not suffice.

Command `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.clean_initialization_audit`
from `/home/haipd/SLR`. Hard timeout30min, monitor≤60s. Outputs
`AS-C19-INITIALIZATION_run.json` and internal TRAIN index partition JSON only.
No large new caches, optimizer state, release checkpoint or fold-training run.
Preserve failures, no silent retry/overwriting or weakened provenance gate.

## Explicit correction before attempt V2

First attempt exited1 at missing-key assertion, before any split/model smoke.
`AS-C19-INITIALIZATION_run.json` retained with traceback and source hash.
The missing set also contains `clip.logit_scale_first_softmax` and
`clip.logit_scale_sec_softmax`. Source `module_clip.py:521–522` initializes both
as scalar ones; source search finds no other references in modeling/module_clip.
V2 explicitly accepts ONLY these two additional constant-initialized tensors,
asserts scalar value1, and separately records them. Three random keys unchanged;
all other unknown/mismatched keys still fatal. No tolerance/provenance relaxation.
Same command now writes `AS-C19-INITIALIZATION-V2_run.json` linked to failed run.
