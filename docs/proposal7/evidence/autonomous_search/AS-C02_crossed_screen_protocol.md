# AS-C02 crossed raw-information screen (registered before training)

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-14
- Verification Status: UNVERIFIED (implementation tested, retrieval runs pending)
- Version Label: crossed-screen-v1

Question: does retaining spatial information from the SAME frozen baseline I3D
permit recoverable ranking improvements beyond pooled/additional-processing controls?
This is one bounded raw-information branch, not a claim that the experiment exposes
all possible hand, face, body, order, or simultaneity information.

Inputs: fixed R0 contextual train/dev features, plus eight uniform windows' 2x2
spatial maps or global pooled vectors. Exact historical extraction batches and
strict pooled parity required. All 7,096 train and 519 dev IDs must be present.
The smoke cache is explicitly rejected by the loader. No test inputs.

Crossed factors:

- R2: small learned raw/text projections followed by masked bidirectional token
  interaction. Two zero-initialized channel coefficients preserve exact R0 at
  initialization. Score scale equals R0's fixed logit scale, avoiding a trivial
  temperature disadvantage for the new branch.
- R3: raw/text projected interactions plus nonlinear pair readout and the AS-C01
  existing-feature readout. Zero final layers preserve exact R0 initialization.
- Five input arms within each regime: spatial information, spatially pooled
  information, train-only shuffled spatial information (aligned dev inputs),
  reprocessed existing R0 features, and zero-input control.
  The existing-feature control retains 32 uniformly sampled valid contextual
  tokens (each 512-vector duplicated to 1024 input dimensions), NOT just the
  temporal mean. This control refinement was made before any R2/R3 training or
  retrieval outcome, to avoid confounding raw access with multi-token access.
- Parameters and module execution are identical across information arms within
  a regime. R3 is larger than R2, so compare each to its OWN capacity controls.
  Different total capacity must not be attributed to raw information.

All arms: seeds 42/1337/2026; 600 updates; batch64 comprising 32 train anchors and
train-only strongest confusers; AdamW lr3e-4, wd.001, betas .9/.999; cosine decay;
float32; clip gradient norm1. Fixed paired positives. Four-direction balanced
contrastive objective. Clean-only text in every arm, as in AS-C01: this screening
deviation from historical random swaps remains explicit. Any survivor needs the
full augmentation-matched fairness campaign before a method claim.

Dev selection every100 steps, mean official bidirectional R1 then mean R5;
initialization eligible. Save all evaluated scores/ranks, initial/selected/last
weights, controls, per-query margins and fixed persistent hard-pair accuracy.
The fixed >=.5pp / >=2-of-3 / positive source-cluster CI / direction and R5/R10
nonregression / persistent improvement / beyond-initialization gates are unchanged.
Never interpret a negative limited readout as proof of unavailable information.

After cache completion and validation, execute:

`PYTHONPATH=shared:. timeout 1800 /home/haipd/miniconda3/bin/python -m methods.information_probe.train_raw_readout`

Expected IDs: `AS-C02-{R2,R3}-{existing,pooled,spatial,shuffled,zero}-s{42,1337,2026}`.
No training command has been run at registration time. Report shared extraction
cost separately from training; later compute claims require isolated timing.
The feature-parity repair and original failed smoke remain separate evidence.
