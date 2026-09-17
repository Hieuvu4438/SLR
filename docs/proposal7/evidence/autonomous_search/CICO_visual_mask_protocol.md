# Visual mask information-flow certificate

2026-09-16; fixed before execution. Source-only, no real data/checkpoints.
Question: does the configured 2D/sum FeatureTransformer permit padded inputs to
affect valid outputs, and are masked-query outputs content-independent zeros?

Extract LayerNorm, QuickGELU, VResidualAttentionBlock, VTransformer and
FeatureTransformer unchanged from pinned module_clip.py, SHA256
8fae51a298872c9d9e07e898a7f09a117cdfbd21a0fc51c2f3a608af2ee6d2d5.
CPU FP32, seed42, batch1, width8, heads2, layers2, four feature slots, first two
valid, last two zero-padded; input channels1024 as source. Output projection6.
CLS mask1, valid mask0, pad mask1, matching loader/bridge polarity.

At fixed shape and positions, replace pad inputs by independent random values
times3. Require unchanged valid outputs within1e-6, and zero pad-input gradient
from a fixed valid-output linear functional within1e-9. Perturb only valid inputs
by fixed random noise and measure masked-output change; measure valid-input
gradient from a fixed masked-output functional. These latter values are evidence
of possible information flow, not prevalence or useful linguistic information.
Use independently random readout vectors, not a LayerNorm-cancelled sum.

Do not change masks, widths, checkpoints or seeds to rescue an outcome. No
retrieval scoring, training, GPU, corpus access or mask-repair method. This
certificate cannot explain C14's observed performance change by itself.
