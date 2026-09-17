# AS-C44 result: raw-readout scope corrected; no method GO

## Material Passport

academic-research-suite / experiment-agent, structural certificate and source
verification; 2026-09-15. ANALYZED overall. AI-assisted code inspection and
synthetic execution, not human linguistic review or a retrieval efficacy study.

## Measured result

The registered CPU certificate completed on its first execution, exit 0,
0.098776 seconds. No dataset, cached feature, checkpoint or raw video was loaded;
zero training updates. Run: [AS-C44-RAW-RELATION_run.json](AS-C44-RAW-RELATION_run.json),
SHA256 `7a184b678aadb75008e78870e6f6324a63bbef48e3e0ccf2c7c4a2b35473dcb1`.
Protocol and all five inspected implementation hashes are embedded in that run.

Both R2 and R3 raw branches are invariant to the registered within-window cell,
window-order and arbitrary flat-slot permutations. Maximum synthetic FP64 error
was 5.551115123125783e-17 (registered tolerance 1e-10). These were nonzero,
content-sensitive outputs: replacing one vector changed both channels by
1.319591e-4 to 3.799575e-4 across the two regimes. This avoids the vacuous
zero-initialization test. It does not replay historical trained weights.

The algebraic reason is visible in the hashed code: a shared pointwise raw
projection commutes with permutations; channel scoring reduces over the raw
axis; the R3 pair branch receives a mean over that axis. There are no raw cell
or window identifiers. Existing contextual video/text inputs were held fixed.
Thus the *post-I3D raw branch* cannot distinguish an ordering of the same vectors.
I3D itself can encode local spatial and temporal relationships inside a vector.

| Input axis | Mixed_5c size | Maximum nominal RF / jump | Adaptive bins | Pooled axial support |
|---|---:|---:|---|---|
| Time, 16 frames | 2 | 99 / 8 | {0,1} | All 16 input indices |
| Height, 224 pixels | 7 | 379 / 32 | {0,1,2,3}, {3,4,5,6} | All 224 indices in each bin |
| Width, 224 pixels | 7 | 379 / 32 | {0,1,2,3}, {3,4,5,6} | All 224 indices in each bin |

Set-valued propagation and independently constructed Boolean incidence agree
at every endpoint. Torch adaptive-pooling basis responses confirm the bins,
including their shared middle cell. Individual final spatial locations have
198–224 axial support indices; the *pooled bins* each cover the full axis.
These are possible architectural dependencies, unioned across channels—not
effective gradients, learned nonzero influence, anatomical segmentation or a
separately enumerated joint three-dimensional voxel-support certificate.

## Validation and execution disclosure

Two new fixture tests passed. The first full-suite invocation omitted unittest's
package root and failed with 30 relative-import errors (exit 1); no code was
changed to address that invocation error. Correcting the root ran only the eight
unittest-style tests, not the function-style suite. The complete check was then:

`PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m pytest methods/information_probe -q`

Result: **79 passed in 1.83s, exit 0**. None of these is a retrieval gate or an
independent human/agent review. No experiment retry, historical-model edit or
active worker remains.

## Decision and limits

AS-C02 remains a negative result for its registered readout and training budget.
It does not directly close explicit recovery of spatial or cross-window
arrangement. Conversely, relabeling its four mixed spatial cells as hand/face
streams and shifting or swapping them would not isolate articulators. The
certificate rejects that intervention interpretation, not the scientific value
of simultaneity. It does not demonstrate that actual PH residual errors arise
from a lost relation, or that semantically distinct videos yield permuted tokens.

No inferential statistics are used: synthetic numerical tolerance is not a
p-value, confidence interval, effect size or estimate of dataset prevalence.
There is no tested candidate, parameter/compute-matched efficacy comparison,
three-seed improvement or second-dataset signature. `method_go=false`.

The associated [primary-source check](AS-C44_sources.md) also prevents promoting
generic stream synchronization or multi-channel attention as novel. RCLI,
generic global composition/order, extra streams and local-evidence rerankers
remain closed under the existing negative registry. No positional-readout,
RF-weighting or synchronization-loss rescue follows from this audit.

Next decision: establish whether documented manual annotation resources can be
linked safely to the existing TRAIN/DEV videos and support the *specific joint
linguistic contrast*, before any relation intervention. Published availability
is not verified local compatibility. Model-generated alignments must not become
expert truth; DEV labels must not become training labels. If this route cannot
provide a distinct open mechanism, switch question rather than tune a closed
family. The search goal remains active; this certificate is progress, not GO.
