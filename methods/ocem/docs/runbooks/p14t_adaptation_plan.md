# P14T adaptation planning

The adaptation planner freezes the IDs and observable CiCo hyperparameters
before pseudo-label generation. It never reads validation or test examples as
adaptation inputs.

## Command

```bash
PYTHONPATH=src python -m ocem features plan-adaptation \
  --train-manifest runs/wp03_phoenix/manifests/train.jsonl \
  --forbidden-manifest runs/wp03_phoenix/manifests/validation.jsonl \
  --forbidden-manifest runs/wp03_phoenix/manifests/test.jsonl \
  --checkpoint ../../artifacts/pretrained/bsl5k.pth.tar \
  --checkpoint-sha256 6430592464a357dfdaa7f31973cb684663237655fdf23f3999608d162167fc6f \
  --class-vocabulary ../../third_party/SLRT/CiCo/I3D_trainer/misc/phoenix2014T/class.txt \
  --seed 0 --holdout-modulus 10 \
  --output runs/wp04_phoenix/adaptation_plan.json
```

The deterministic assignment uses SHA-256 over the version, seed, and sample
ID. The resulting 6,349/747 train/holdout partition is entirely inside the
7,096-example official P14T train split. Its overlap with the 519 validation
and 642 test IDs is zero.

The plan pins the released threshold and temporal suppression behavior rather
than inheriting undocumented defaults. It also records the trainer defaults,
even though the released schedule milestones occur after its default 15-epoch
budget.

## Real I3D smoke parity

The shared deterministic extractor loaded the hashed Oxford checkpoint with
strict state-dict matching and evaluated the 16-frame train video
`01April_2011_Friday_tagesschau-3379`. Its `[1,1024]` output was bit-identical
to the audited cache (`max_abs_error=0`). This validates the current
checkpoint/layer/preprocessing route; it is not an adaptation result.

## Remaining guard

The plan is ready for pseudo-label generation but not adaptation training. A
manifest-indexed pseudo-clip dataset must replace the missing/hard-coded P14T
trainer inputs, pass a one-batch forward/loss/gradient/update test, and prove
that it consumes only plan-assigned train/holdout IDs.
