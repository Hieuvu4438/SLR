# Recovered-frame retention: bounded Stage0 protocol

2026-09-22, registered before aggregate metadata counts are examined.
Question Q19: does the64-window cap actually omit any retained source frames in
the recovered adapted PH inputs? Separately, which preprocessing stage accounts
for the13 CTC-ineligible TRAIN examples discovered in Cycle7?

Source-derived decomposition, not a new sampling method:

1. Raw decoded indices `R = range(decoded_frames)`.
2. Native pre-filter grid `G = range(0,N,interval)`, where interval is1 when
   N≤300, otherwise `max(2,ceil(N/300))`.
3. Stored retained indices K must be a subset of G. `G\K` is support-filter
   removal, distinct from intended subsampling `R\G`.
4. U is the union of recorded original-frame indices in all selected windows.
   `K\U` is retained-frame omission caused by subsequent window selection.

Exact falsification: if `K\U` is empty for all7615 examples, reject the claim
that the64-window cap omits retained frames **in these recovered inputs**. No
post-hoc near-coverage threshold. If nonempty, retain exact affected IDs for
source attribution; do not infer semantic harm or implement a sampler.

For each of the13 ineligible TRAIN cases, report N, |G|, |K|, number of selected
windows, and necessary CTC steps. This separates intrinsically short input from
frame filtering. No automatic change to head, masks, labels or corpus membership.

Source hashes/metadata inventory must match the completed recovery audit. This
does not compare deleted historical inputs, assess frame quality or prove what
the encoders retain. No model inference, raw-video decode, GPU, TEST, annotation
collection, changed relevance, training or active-job check. The prior closures
on sampling/OT/extra observations remain binding under either outcome.
