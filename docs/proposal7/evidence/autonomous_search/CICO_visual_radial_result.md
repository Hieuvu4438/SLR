# CiCo visual radial feasibility: local ambiguity, not a semantic bottleneck

2026-09-16. ANALYZED; AI-assisted academic-research-suite source/claim checks.
This follows the [text radial check](CICO_radial_contract_result.md), whose
native-precision inverse remains unsuccessful. No method GO or training run.

## New evidence

The actual visual projection is768→512, so the text branch's square-projection
inverse cannot simply be transferred. We checked the feasible input set of the
final LayerNorm and projection using native seed42 parameters and directions
from the first real video slot of the first256 TRAIN rows, fixed in advance.

| Output magnitude relative to stored token | Feasible at prescribed LayerNorm norm | Maximum minimum-required squared norm /768 |
|---|---:|---:|
|0.9|256/256|0.780435|
|1.1|155/256|1.106714|

The stacked513×768 linear constraint matrix has rank513 and condition4498.745.
Maximum linear residual is4.30e−14. Thus155 of these256 directions admit BOTH
prespecified magnitudes in the ideal final-layer input domain. The statement is
about directions from a fixed TRAIN prefix, not a random-population estimate.
It does not identify actual pairs of videos with identical normalized features.

The row0 end-to-end local construction was **not performed**: its two endpoints
did not both satisfy the prerequisite. The protocol explicitly prohibited
substituting another row, so neither native-dtype nor FP64 row0 forward parity
is claimed. The numerical feasibility result is preserved without changing its
scales, threshold or sample. Program exit0 means the planned check completed.

Artifacts: [locked protocol](CICO_visual_radial_protocol.md),
[run](CICO-VISUAL-RADIAL.json),
[script](../../../../methods/information_probe/visual_radial_contract.py).

## Why this is a valid local distinction

Let z=(gamma*u+beta)P, sum(u)=0, and ||u||²<D. A zero-mean pre-LayerNorm input
h=u sqrt(epsilon/(1−||u||²/D)) produces u before its affine transform in exact
arithmetic. The minimum-norm solution of

`[(diag(gamma)P)^T; ones^T/sqrt(D)] u = [z−betaP;0]`

therefore establishes feasibility when its norm is below the LayerNorm bound.
An orthogonal nullspace component can raise its norm to the fixed target
D*(1−1e−5) without altering z or the zero-mean condition. Two positive multiples
of the same output then have the same unit direction despite differing radii.
This is a function-class observation, not a novel inversion or retrieval method.

Two independent small synthetic tests verify the minimum-norm/nullspace
construction, equal hidden variance, exact LayerNorm output and infeasibility
outside the ball. Combined with the three text-contract tests:5passed.
These fixtures do not substitute for the skipped native-checkpoint row0 example.

## Decision

The visual local map permits radial ambiguity; unlike the ideal square text
case, one cannot dismiss visual norm loss using that affine inverse. But the
preceding transformer constrains which hidden states are actually reachable.
This check supplies no evidence that two real videos occupy the constructed
states, that magnitude discriminates meanings, or that retaining it improves
official ranks. It therefore does not justify a norm gate, a cosine-removal
sweep, a precision-training rescue or reopening closed representation methods.

Both radial turns are now complete. Move to a different mechanism. Do not run
more scale/row/precision variants to turn this local feasibility result into a
passing example. A new method still requires an admissible measured bottleneck,
material novelty and all unchanged controlled GO gates. No new Q38, Proposal8,
global information ceiling or global research barrier follows.

## Scope and verification

Only an existing TRAIN cache and historically DEV-selected seed42 checkpoint
were read. No new selection, caption labels, raw videos, encoder/scorer forward,
optimizer, GPU, DEV/TEST cache, SEDS assets or upstream edits. Source pin remains
SLRT38a4f7b00da7a858d59b7fabe5093876a84db8e0. Hashes are in the run JSON.
The preceding text certificate's six source fingerprints and precision
validation's parent-result hash were rechecked and matched.

Before this check, C22/C23 were inspected and ruled out repeating the already
completed text-omission/restoration audit. C24/C25 likewise already cover
soft-expectation derivative signs and endpoint/entropy controls. These reads
prevent duplicate work; they are not new empirical findings.
