# AS-C44: raw-probe relation and intervention-validity certificate

## Material Passport

academic-research-suite / experiment-agent, code/structural audit; 2026-09-15.
ANALYZED protocol, not a method pilot. Locked before executing the certificate.

Question: can AS-C02's cached four spatial vectors be interpreted as isolated
articulators, and did its readout directly test their temporal/spatial arrangement?

Decision consequence: if the readout is permutation-invariant, its negative result
must not be used to close explicit relational recovery. If final spatial bins have
overlapping whole-input structural support, do not use post-encoder cell swaps or
time shifts as isolated hand/face interventions. Neither result licenses a generic
positional encoder, multi-stream fusion, synchronization loss or closed relation
method as a new contribution.

Locked checks, no PH examples, checkpoints, cached tensors or split data loaded:

1. Inspect original `raw_readout.py`, `scoring.py`, `cache_raw.py`, the actual I3D
   implementation, and extraction recipe; hash all five sources.
2. Instantiate the existing R2/R3 readouts with deterministic synthetic tensors,
   FP64 CPU. Set synthetic R2 scales and R3 final weights nonzero so zero-initialized
   residuals cannot make an invariance test vacuous. No historical weights edited.
3. Check three raw-token permutations: spatial cells within each of eight windows,
   window order, arbitrary flattened order. Keep existing video/text inputs fixed.
   The mathematical invariant permits floating-reduction differences; report max
   error and assert≤1e−10. Replacing one raw vector must change outputs>1e−12.
4. Propagate structural support along each input axis through the instantiated
   I3D operations, using actual kernel/stride/SAME-padding. Input16×224×224 from
   the pinned extraction recipe. Report per-layer union-of-channel support and
   nominal maximum-path receptive fields; do not call these effective gradients.
5. Independently propagate Boolean incidence matrices and compare with set-valued
   dependency propagation. Validate adaptive pooling bins against torch's basis
   responses. Output is the actual cache hook's adaptive(1,2,2) shape.

No retrieval result, dataset prevalence, information ceiling, valid linguistic
counterfactual, novel method or global exhaustion can be inferred. No patch to the
old probe or new training follows from this audit. Strong controls and expert-valid
contrasts remain prerequisites for a different open candidate.

Command: `PYTHONPATH=shared:. timeout 120s /home/haipd/miniconda3/bin/python -m methods.information_probe.raw_relation_certificate`.
Output: `AS-C44-RAW-RELATION_run.json`, current evidence directory; refuse overwrite.
Monitor actual process/exit code; CPU-only, no training. Autonomous loop authorizes
the local diagnostic; skill roles inline, no subagents.
