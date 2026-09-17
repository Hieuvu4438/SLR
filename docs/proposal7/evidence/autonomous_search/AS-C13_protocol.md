# AS-C13 — aligned pre/post-context linear predictability

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run (autonomous design authority from governing user request)
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (preregistered before execution)
- Version Label: AS-C13-v1

## Question and adequacy

Q07: how much of each normalized visual representation is linearly predictable
from the other at the same canonical window, and does replacing contextual
tokens by a train-fitted prediction retain retrieval performance?
Unlike the earlier residual classifiers, reconstruction has a continuous,
unsaturated target. This tests LINEAR PREDICTABILITY, not semantic sufficiency,
invertibility, an information-theoretic bound, or a novel method. A failed map
cannot prove that the contextual encoder discards useful information. A positive
reverse map can demonstrate retention of a linearly decodable component only.

## Fixed design

- PH train7096/dev519 only; R0 seed42 frozen cache and canonical fused I3D inputs.
- Assert manifest IDs, valid masks and slot alignment (skip contextual CLS).
- Normalize each pre-context1024D and contextual512D vector. Each sequence has
  equal mass; its valid positions have equal conditional mass. Padding excluded
  from fitting and reconstruction metrics.
- Fit affine ridge in both directions using CPU float64 centered moments.
  Fixed penalty `0.01 * trace(Cxx) / input_dimension`; no sweep, selector or
  dev-fitted quantities. Intercept is unpenalized.
- Null: fixed seed42 random permutation of all valid contextual tokens before
  fitting both maps. This destroys paired correspondence, not a guaranteed
  sequence-level derangement; occasional same-token/video pairings can remain.
  Weight remains attached to the original input slot. Null has the same map
  sizes, fitting operation and number of observations.
- Reconstruction on original aligned train and dev pairs: equal-sequence
  weighted squared error, error relative to the aligned TRAIN target mean,
  explained fraction `1 - error / mean_predictor_error`, cosine, per-sequence
  distributions. Report aligned, shuffled, and mean maps in both directions.
- Full-gallery retrieval: identity, aligned pre→post prediction, shuffled
  pre→post prediction, train contextual mean, and post→pre→post round trip
  (normalize the reconstructed pre tokens before the forward map).
  All variants keep ORIGINAL contextual CLS/pad tokens; replace only valid
  visual tokens. Text and scorer unchanged. Thus these are hybrid diagnostics,
  not transformer-free deployment methods. Inner-softmax padding effects remain.
- Identity channel tolerance2e-5 and exact per-query rank parity before any
  interpretation. Disclose official T2V/V2T R1/R5/R10, persistent rank changes.
- Cache only small map/metric/score artifacts; pre-context inputs held in RAM.
  Every variant reported; no three-seed pilot or GO claim from this experiment.

## Decision rules and limits

Strong reconstruction with poor retrieval indicates the recovered component
does not suffice under this substitution/scorer; it does not locate linguistic
information. Train/dev reconstruction discrepancy tests transfer of this probe,
not its universal power. A forward map approaching identity would motivate a
controlled context-contribution diagnosis, not a distillation proposal.
No threshold here relaxes the governing GO gates. No architectural candidate
from a reconstruction negative alone. Repeated PH-dev exploration is disclosed.

## Execution

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.retention_probe`
from `/home/haipd/SLR`; hard timeout30minutes, progress checks ≤60seconds.
Monitor run JSON and process/session. Expected outputs: `AS-C13-RETENTION_run.json`,
five score/metric pairs and `AS-C13-maps.pt`. On failure preserve JSON/traceback;
report any implementation correction before a separately identified retry.
