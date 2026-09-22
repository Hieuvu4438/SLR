# GCN training-batch normalization screen

2026-09-23, Cycle19; base HEAD53b5f986d74cfeb5f4cc83649eeb44241cab6ea7.
**REJECTED as a current method lead:** eliminating training-batch BatchNorm
coupling cannot explain an improvement over a control that already disables
that coupling. This is a bounded source/fixture result, not a retrieval result.

## Question, prediction and falsification

[H] Updating pose GCN weights may accidentally enable training-batch statistics,
making an individual video's representation depend on its batch companions.
If present, a normalization intervention might address an actual computational
mechanism without semantic annotation. Required first prediction: the current
GCN control must execute BatchNorm using batch statistics. If the actual policy
keeps tracked-statistics BN in eval, reject this explanation before any model
training or normalization sweep.

[V] `third_party/SEDS/modules/modeling_gcn.py` contains data BN, graph-block BN
and temporal-block BN. Mere module presence is insufficient. The relevant
configuration route is `configure_masked_pose(..., control=True)`, not the
similarly documented but inactive `enable_gcn_adaptation` LoRA route.

1. `research/slret_goal_v2/tools/train_seds_extended.py:630` selects the masked
   clean-control policy. Its per-step loop calls `wrapped.train()` and then
   `training_modes(model, active)` before the loss forward.
2. `methods/seds_adaptation/masked_pose.py:112` makes GCN embed weights trainable
   but obtains `active=[model.fusion]` from the selective policy. It does not
   activate GCN training mode. Enabling parameter gradients is not enabling BN
   batch-statistics estimation.
3. `methods/seds_adaptation/train_policies.py:31` recursively calls `model.eval()`,
   sets only the top-level `training` flag for native loss-return behavior, then
   enables each active module. Thus the earlier recursive train call does not
   leave the GCN in train mode.
4. The three retained horizon3 run reports record `masked_pose=control`,
   `policy=fusion`, `gcn_lora=false`, `gcn_long_horizon=true`, and native fusion
   `gloss_atten`. Its `Gloss_Fusion_Transformer` implementation uses LayerNorm;
   the BatchNorm in `module_cross.Conv_feature_fusion` is not this selected
   fusion class. Searching all BN symbols without tracing selection would
   misattribute an inactive path.

These are current-source facts. Historical execution qualification follows.

## CPU counterfactual checks

New [fixtures](../tests/test_gcn_batch_independence.py) AST-extract the actual
native `TemporalConvNetBlock` class, instantiate it with four channels on CPU,
and apply the current clean-control policy. No native CUDA constructor, trained
checkpoint, data sample, video, feature cache or retrieval score is loaded.

- Same held sample and temporal extent, different companions: outputs agree
  within1e-12 absolute tolerance under selective eval; BN buffers are unchanged.
- Positive control with the same block in train mode: companion change produces
  an output difference greater than0.01; the BN batch counter advances twice.
- Under eval BN, a synthetic loss yields finite nonzero encoder gradients and
  changes a convolution weight while all buffers stay exactly unchanged.

First execution: five passed, one failed. Both positive-control outputs were
clipped to zero by the native final ReLU because both companion levels exceeded
the held input. The fixture was corrected from companion levels1/20 to0/20;
no source implementation, threshold or empirical dataset result changed.
This is disclosed fixture development, not a preregistered efficacy experiment.

Final command, exit0, **6 tests passed**:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/haipd/miniconda3/bin/python -m pytest -q docs/codex_slret_research/tests/test_gcn_batch_independence.py research/slret_goal_v2/tests/test_peft_setup.py research/slret_goal_v2/tests/test_train_policies.py
```

This combines three new block/policy tests and three existing policy tests.
It is not a full GCN forward, GPU/DDP parity check or repository-wide test suite.

## Provenance and read scope

The completed run reports inspected are under
`artifacts/slret_goal_v2/seds-gcn-horizon3{,-seed1337,-seed2026}-001/run.json`.
Their stored `train_policies.py` hash matches the current file. Their stored
`masked_pose.py` and extended-trainer hashes do **not** match current files.
Trying `git show` for `masked_pose.py` at the seed42 report's recorded commit
`e3f53b3762a93b3cbe7a2854832654f814f66338` fails because that file is absent
at that revision. Do not claim an exact historical source reconstruction.
No historical file or user's working changes were overwritten to repair this.

Current material hashes (source selections read, no human-read attestation):

| File | SHA256 |
|---|---|
| methods/seds_adaptation/train_policies.py | ec3e9c4b52340422244ae651c9567292bee8ce08e5a728f3df3b469dc4358f01 |
| methods/seds_adaptation/masked_pose.py | d798eccdde27bb59ca43a38c65839215e6a6722169c2977d3cc2a76b071ccf5b |
| research/slret_goal_v2/tools/train_seds_extended.py | 57d2a3822bb7e8e647fa977c934c594028fd9e18e3fe42d5a45dc8ac2830492b |
| third_party/SEDS/modules/modeling_gcn.py | bfbaadcb3154bd23519a85420d155010ce4cdc258cbcbfa92e67536f18e79bd6 |
| third_party/SEDS/modules/module_fusionencoder.py | ae25bc1738ae4b594430e62fa430a52359ca623b5a2ae90d62416ead02068e26 |
| docs/codex_slret_research/tests/test_gcn_batch_independence.py | 447b06b212081a8418531b1197f5f30cb95cb56ef0a6866163c270589f0c2073 |

Runtime: Python3.13 environment, torch2.11.0+cu128 package; fixtures execute
on CPU. No GPU cost or model throughput was measured.

## Adversarial review and limits

ARS checkpoints, performed inline by one assistant, not independent reviewers:

- Scoping: the strongest alternative is that selective eval already removes
  batch-statistics coupling. Inspect call order, not just parameter flags. Done.
- Evidence: current source hashes differ from two historical sources. Restrict
  the result to current executable policy plus compatible historical config
  records; no claim of exact historical runtime verification.
- Final claim: rejecting this cause does not prove statistical optimality of
  frozen BN, absence of inherited pretraining bias, or correctness of the whole
  model. No retrieval gain and no novel normalization method follows.

Four task perspectives: novelty has no supported new mechanism; experimental
evidence is a small computational fixture; sign-language meaning is not inferred;
retrieval gradients can still depend on other examples through the contrastive
loss even when pre-loss representations do not. That last dependency is not
BN leakage. Within-video temporal padding/convolution effects are also not
excluded by same-extent companion substitution.

No source-only inference is promoted to a historical reproduced result. No
aggregate error correlation, p-value, semantic label or performance ceiling is
used. The fixture is a positive-control check for this operator, not evidence
that natural errors are irreducible. Do not widen the rejection to all possible
normalization research; do not launch a BN-recalibration or norm-type sweep
from this failed premise.

## Decision consequence

Stop the current training-batch BN contamination lead. No candidate satisfies
the measured-bottleneck gate here. This changes the next action: no costly
activation dump or pilot is needed for this claim; discovery must identify a
different active mechanism. No annotation, model training, TEST access, deferred
job check or external contact. ARS supplied counterevidence checks and source
qualification. Main method and SOTA objectives remain active and incomplete.
