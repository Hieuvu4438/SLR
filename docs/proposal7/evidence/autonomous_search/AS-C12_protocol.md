# AS-C12: directional gradient and collision-support diagnostic

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: run
- Date: 2026-09-15
- Verification Status: preregistered, not a method or full-encoder gradient claim.

Q14/Q16 optimization layer, distinct from C11 feature metric. Use fixed R0 train
features only. For each seed42/1337/2026, first four disjoint512-example batches
from a deterministic permutation of7096 train rows (6144 batch exposures total;
rows overlap across seeds). Compare clean captions and exact deployed epoch0
augmentation. Re-encode augmented text in original128-row order to retain the
established numerical contract; require exact clean parity first.

Compute A from clean text; B from clean or augmented text. Losses are
L_V=(CE(A)+CE(B))/2 and L_T=(CE(A.T)+CE(B.T))/2. Differentiate each loss with
respect to the shared contextual VIDEO tokens only. Report global cosine,
fraction of examples with negative gradient cosine, norms, and squared-norm
mass on examples whose clean deployed text has duplicate input IDs inside
the current batch. Also report global-train duplicate membership. These are
interface sensitivities, NOT parameter gradients: shared encoder Jacobians may
align/cancel them. No invented gradient-surgery or RPCA candidate follows.

Full512 negative pool per diagnostic batch; no hard-negative miner. No dev
labels, tuning, optimizer, parameter update, test or SEDS. Report all24 batch
conditions. No inferential significance from overlapping examples or seeds of
augmentation. Hard timeout600seconds. Exact batch indices, losses and summaries
saved; no checkpoint mutation. Absence of conflict at this interface does not
prove absence of conflict in parameters or along training.
