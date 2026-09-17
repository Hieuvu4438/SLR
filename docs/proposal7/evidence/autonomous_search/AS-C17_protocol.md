# AS-C17 — fixed-I3D temporal-view sensitivity

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C17-v1

Q19: is canonical64-window sampling sensitive to nearby legal choices from the
SAME frozen I3D streams? This is acquisition diagnosis, not a new backbone,
active acquisition, sampling-consistency loss, partial alignment or new method.

One frozen R0 seed42, PH dev519 full gallery. Shared CiCoFeatureDataset with
feature_len64, alpha.9, existing `jittered_view`, seeds42/1337/2026 fixed. Each
view preserves sorted unique indices, count, valid mask and canonical positions;
each chosen dense index is within1 of its canonical counterpart. When no legal
alternative exists, retain identity and disclose the count. This alters chosen
overlapping I3D windows; it is NOT a validated signing-rate/semantic intervention.

Execute canonical encoding THREE times and each jittered encoding once. Repeated
canonical passes must be identical to each other and the existing dev token
cache; score channels≤2e-5 and exact ranks. Frozen text/cache/scorer unchanged.
Three-repeat score mean is the equal-pass control for the three-jitter score
mean. Report each jitter separately, both means, directional R1/R5/R10,
persistent ranks, original index displacement and contextual token changes.
No dev selector, new parameters, training, changed positives or test access.

All views share one trained backbone; the three view seeds are NOT independent
training seeds. A fixed view mean is standard test-time averaging, not a novel
method. If it exceeds the equal-pass control by≥.5pp mean R1 without losing
>.25pp in either direction, it is only an exploratory mechanism lead requiring
the full unchanged GO contract, not automatic candidate survival. Otherwise do
not tune view seeds or frame count to rescue it. This check does not exhaust
all temporal acquisition mechanisms.

Command `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.temporal_view_probe`
from `/home/haipd/SLR`. Hard timeout30min, monitor session≤60s. Outputs
`AS-C17-VIEWS_run.json`, per-view/mean metric files and small score arrays.
No new feature caches; use RAM. Preserve failures, no overwrite or silent retry.
