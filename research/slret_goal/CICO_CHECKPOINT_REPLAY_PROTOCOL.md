# Independent-process DEV checkpoint reconstruction gate

2026-09-18 preregistration; ARS reproducibility check. This is checkpoint
serialization/evaluation reproducibility, NOT independent scientific confirmation.
Run only after the fixed replication queue and six paired CPU audits complete.
No concurrent training GPU job; no TEST, optimizer updates or selection changes.

Reconstruct all12 final checkpoints: PH/CSL×3seeds×2moment arms. No favorable
checkpoint subset. Verify source-run/audit/checkpoint/config/feature identities;
load full model tensors strictly into unchanged shared CiCo/ELSC-baseline wrapper.
Same Torch2.11/CUDA runtime, model dtypes, evalbatch128, scorerblock128,
PH519singleton andCSL1077×797 grouped galleries. Fresh process, model.eval,
same deterministic seed settings and original data/evaluator code. Use separate
run IDs/output paths and retain all score matrices.

Require score bit-exact equality and exact saved metrics/ranks/ID hashes at
each endpoint. No tolerance weakening if a gate fails; investigate serialization,
environment or evaluator before final evaluation. This is a deterministic
replay, so no 5%/10% default relative tolerance. No runtime/latency comparison
claim. Missing/failed run => gate incomplete, never average over survivors.

Budget <=300s and<=100MiB output, charged after replication to remaining
discovery reserve. Registered7000s queue leaves>=1055s before itsactualusage;
the300s bound fits. Campaign44GiB cap/15GiB diskreserve still apply andmustbe
checked before launch. Prepared tool is not currently running while queue lives.
