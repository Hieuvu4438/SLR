# Retrieval-gradient RGB feasibility — 2026-09-18 preregistration

Purpose: measure whether a stronger baseline can update the existing RGB I3D
with the actual SEDS retrieval loss under the local memory/time envelope.
Not a novel method/candidate admission and not an efficacy experiment.

Evidence motivating this check: adapted fusion improves over its jointly trained
RGB branch, but most fusion failures are shared RGB/pose errors; native and
FP32-moment continuation fail the+0.5pp improvement gate. SOURCE_VERIFIED RGB
features are frozen offline while context/pose train. This does NOT establish
that lost RGB information causes those errors. Direct retrieval adaptation is
a testable stronger-control route before inventing extra objectives/streams.

Collision scope: same I3D architecture, weights, spatial preprocessing, full
selected windows and official TRAIN captions; no replacement backbone/scale,
raw-grid readout, residual adapter, lexical teacher, local evidence, temporal
consistency, generation, miner or scorer change. AS-C44 explicitly studies
post-I3D permutation-invariant readouts, not supervised updates inside I3D.
Generic end-to-end fine-tuning itself is established practice, not novelty.
Gradient caching attribution: Gao et al., RepL4NLP2021,
https://aclanthology.org/2021.repl4nlp-1.31/ (primary abstract/algorithm description
inspected; no claim of a newly invented memory-saving algorithm).

CPU mechanical tests first. Then <=300s GPU diagnostic: first full32-example
batch/order from native continuationcontrol001; SEDS release weights; native
loss/augmentation, no optimizer. Differentiate loss w.r.t. the RGB feature leaf.
For the firsttwo videos, decode exact recorded raw frames/windows, regenerate
all features in microbatches8 and compare with saved inputs(maxabs<=1e-4).
Unfreeze only existing Mixed_5b/Mixed_5c parameters, with BatchNorm running
statistics and dropout kept in eval mode; no new objective or representation.
Backpropagate cached representation gradients through every window of those
two videos. On the first8 windows compare direct batch VJP with two chunks4
(relative L2<=1e-4 or absolute<=1e-8 for tiny gradients).

Gate: feature parity, finite nonzero RGB gradients, finite nonzero I3D gradients,
chunk/full gradient equivalence, actual time/VRAM plus conservative full7096
projection. Zero updates/checkpoints; <=32MiB diagnosticoutput, preserve15GiB
campaigncap/free-space reserve. A later222update pilot requires separate
preregistration, storage revision and matched controls; this check does not
authorize it or imply recall gains. Report startup/decoding/backward separately;
two-video throughput is provisional, not a measured full-epoch budget.

Startup repair2026-09-18: attempt001 failed before model loading because native
SEDS lacked cv2 (0.799s; zero updates). Preserve its report/ledger. Retry002 adds
only opencv-python-headless4.11.0.86 with pip --no-deps; keep NumPy1.26.4 and
torch2.3.1 unchanged. Extraction used OpenCV4.13.0/base torch2.11; do not assume
cross-environment parity. The original maxabs<=1e-4 gate stays mandatory and
versions are recorded. Package source: https://pypi.org/project/opencv-python-headless/4.11.0.86/.
No increase in GPU timeout, output allowance or efficacy scope.

Environment repair executed: slow indexed wheel download cancelled before
installation, then identical version obtained from official files.pythonhosted.org
with PyPI SHA256 verification:
`0e0a27c19dd1f40ddff94976cfe43066fbbe9dfbb2ec1907d66c19caef42a57b`.
NumPy1.26.4/torch2.3.1+cu121 rechecked unchanged; cv2 imports4.11.0.
Projection uses actual TRAIN metadata census, worst observed per-window RGB
forward+replay and per-decoded-frame rates, adds50% RGB safety margin plus
completed native-head control wall time. It excludes new DEV extraction and
new I3D optimizer overhead, explicitly not an admission estimate on its own.

Attempt002 terminal FAILED original feature gate: firstvideo maxabs0.0003829002
>0.0001. Native first-batch losses exactly match control and RGB-input gradient
is finite/nonzero; no I3D backward/update occurred. Do not relax tolerance.
CPU decoded/preprocessed frames match bytewise across OpenCV4.11/4.13:
`ed8c2ad0e3544c502d97c307e25fda6944957245042523d3770482d168bc1070`.
Admit a localization diagnostic, `rgb_runtime_parity.py`, firstsame64windows,
base and native runtimes, two cuDNN deterministic settings. Compare shared
extractor, differentiable helper and savedfeatures. <=120s each, tiny JSON only,
no losses/updates or DEV/TEST. Tests a specific runtime confound, not newmethod.

Localization results: native2.3.1 has maxabs0.0003829002 in both cuDNN
deterministic settings; helper/shared match bitexact. Base2.11.0 reproduces
savedfeatures bitexact under both settings, with identical framehash. Thus use
original RGB runtime for encoder VJP and native runtime for SEDS head; do not
silently upgrade SEDS or loosen parity. This is a mixed-runtime composition,
not proof of equivalence to training everything in either one environment.

Activation amendment before encoder-backward outcomes: cached native batch has
26/32 video gradients exactly zero, including the originally chosen firsttwo.
Their zero parameter VJP would be expected, not a graph failure. For the new
activation check select the firsttwo indices with nonzero gradient (3,7), without
using retrieval ranks or tuning outcomes. Keep zeros/all32norms in report; this
selection cannot estimate average gradient exposure or efficacy. On index3,
the first8window gradient is nonzero, so the preregistered chunk test is meaningful.
`rgb_cached_vjp.py` reads the preserved002 head gradients (source loss exactly
matched control) and original storedfeatures. <=300s, tiny JSON, no optimizer,
no extra cachedfeatures, unchanged tolerances and BN/dropout controls. Estimate
excludes cross-process transfer and must not alone admit long training.

CachedVJP001 completed: bothfeatures bitexact,36/36tail tensors havefinite
nonzero gradients, BNunchanged. IMPORTANT:8-vs4chunk test passed ONLY its
absolute branch (2.327e-9); relative error2.5726% is not strong equivalence.
No arbitrary-microbatch precision claim. Actual replay and extraction bothuse8.
Admit one stronger diagnostic<=120s, no updates: first16windows of activated
index3, normalize actual cachedVJP to unitL2, compare direct retained graph
(concatenate two forwards of8) to separate replay backwards (same two8).
Require bitexactfeatures and relative parameter-gradient L2<=1e-4, with NO
tiny-absolute escape. This isolates graph caching from changing cuDNN batching;
failure stops full-training admission. Keep earlier2.57% result visible.
