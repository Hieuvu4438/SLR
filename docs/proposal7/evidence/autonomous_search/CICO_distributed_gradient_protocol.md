# CiCo distributed-gradient contract: fixed source fixture

2026-09-16, registered before execution. Source-only CPU FP64 check, no benchmark
data, checkpoint, GPU or training. Hypothesis: full global loss on each rank plus
local-slice gather backward and DDP gradient averaging yields encoder gradients
1/W of the single-process global-batch reference, but unchanged logit-scale
gradient. This is a reproduction concern, not novel optimization or causal harm.

Use pinned local AllGather and CrossEn classes plus flip_similarity_softmax AST.
Emulate only forward communication by copying predetermined feature shards;
execute the original gather backward. Arithmetic average of replica parameter
gradients represents standard DDP reduction; no claim of actual multiprocess or
NCCL execution. Fixed seed42, B8, visual2/text3 tokens, input4/output5 dimensions,
two shared linear encoders, separate scalar logit scale initialized .7. Augmented
text is a fixed additional random tensor. All masks valid; four-CE balance loss.
World sizes1,2,4 with identical global inputs, parameters and loss.

Controls: multiply gathered-feature gradients by W (should restore all reference
parameter gradients); multiply entire loss by W (restores encoders but multiplies
temperature gradient). Compare all coordinates, tolerance1e-10. No optimizer
updates or parameter search. Failed assertions remain failure, not retuning.

Separately inspect whether the existing local baseline enables distributed
gather. No correction campaign if exposure absent; no extrapolation to the
unknown historical author run or to Adam update magnitude from gradient ratio.
