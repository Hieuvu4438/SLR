## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (registered before execution)
- Version Label: AS-C33-v1

## Question / existing-prior boundary

AS-C32 fixed3-model score ensemble improves mean R1 by2.023121pp with conditional
95% CI[.465506,3.619048]. Does a SINGLE fixed equal-weight parameter average
retain that gain at one-model inference cost? This is a closest-prior/stronger
control test, explicitly NOT a new method. Parameter averaging is already the
mechanism of [Model Soups](https://proceedings.mlr.press/v162/wortsman22a.html).
No learned mixture weights, layer selection, greedy soup, precision sweep or
new optimization. The three existing selected checkpoint files stay untouched.

## Two fixed variants and provenance

Repeat3seed42 parameter average is the identity control. Uniform3weights uses
seeds42/1337/2026 exactly as AS-C32. Require identical state key sets/shapes/dtypes.
Floating tensors: accumulate in float64, divide by3, cast to original dtype.
Nonfloating buffers must be identical, then copy. No normalization or learned
alignment. Logit-scale parameters averaged in their stored log coordinates.
Hash ordered tensor names/dtypes/shapes/raw bytes for reconstructable provenance;
do not write a new large checkpoint for this diagnostic.

Validate original checkpoint hashes against AS-C32; repeat3 ALL state tensors
must exactly equal seed42. Reload native-precision encoder, encode original
DEV519 features/captions in128-sized batches, original feature_len64,alpha.9,
text_len32, no augmentation. Repeat3 video/text cache tensors/masks/CLS and full
original bridge scores must be bitwise equal to R0. If not, preserve failure,
diagnose without altering the parent average/gates. No test access.

Uniform weights evaluate SAME independent-query scorer and full gallery. One
model has unchanged149724163parameter count and architecture. Two comparisons:
original single42 and AS-C32 uniform SCORE ensemble. Report R1/R5/R10, persistent
ranks, conditional tie-aware315-source-prefix cluster CI10000draws using the
AS-C32 function. A one-off score ensemble/soup is not3independent method seeds.

## Decision

Descriptive gain over strongest single model, and prespecified ensemble-retention
gate: mean R1 no worse than score ensemble by>.25pp, neither direction loses
>.25pp R1, R5/R10 losses<=.5pp. This gate is diagnostic only. If it passes,
use soup as a stronger single-model baseline, not a novelty claim. If it fails,
do not tune layers/mixtures/temperature to rescue this fixed prior-art control.
No empirical result alone establishes flatness as its cause or a new method.

## Execution

    PYTHONPATH=shared:. timeout 300 /home/haipd/miniconda3/bin/python -m methods.information_probe.weight_soup_probe

Output AS-C33-SOUP_run.json and two AS-C33 score matrices under phase2 artifacts.
No overwrite. Expected tens of seconds,<=4GB GPU;300s timeout and active output/
process monitoring. All attempts retained, unit tests before run,11/11 fallacy
scan after. Pretrained/checkpoint-dependent provenance and dev-selection limits
remain unchanged. Original GO criteria and closed research families unchanged.
