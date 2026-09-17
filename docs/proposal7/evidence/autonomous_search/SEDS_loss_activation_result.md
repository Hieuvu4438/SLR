# SEDS loss-activation gate: optional KL is not the released recipe

2026-09-16. academic-research-suite/source verification inline. AI-assisted,
ANALYZED; no human review, no SEDS assets accessed, no method GO.

## Decision-changing finding

The top3/top5 cross-stream KL code inspected earlier is **optional and absent
from all three supplied training recipes**. It must not be presented as a
measured weakness of the released PH/How2Sign/CSL training configuration.
Do not launch a top-k KL repair, support/teacher swap or synthetic KL-gradient
campaign on that attribution. No claim about unpublished author commands.

Pinned code:434e3f714fcb6a7d1f4001fb9a246bbd93ec0246, local worktree unchanged.
[Parser/recipe certificate](SEDS-LOSS-ACTIVATION.json),
[executable](../../../../methods/information_probe/seds_loss_activation.py).

| Inspected control | Result |
|---|---|
| `--rgb_pose_kl` | store_true, defaultFalse; omitted in all3 launchers |
| Explicit KL flag positive control | ParsedTrue |
| `--rgb_pose_match` | Present in all3 launchers |
| `--rgb_pose_match_loss` | .4 in all3 launchers |
| KL construction/call | Guarded by rgb_pose_kl at modeling.py:199/516 |
| Constructor freeze_exfusion | Required attribute at modeling.py:99, absent from parser namespace |

The actual requested auxiliary objective is Pose–RGB clip-index matching.
In modeling.py:523–539, the code forms cross-video token similarities,
softmaxes over one token axis, multiplies by a token-index identity mask,
sums the surviving diagonal contributions, and applies the learned logit scale.
The forward loss then uses bidirectional CrossEn and coefficient.4. Text/video
alignment uses fusion, pose and RGB branches. This is a source interpretation,
not a fully executed training trace; startup issues below prevent a claim that
the unmodified release has been reproduced here.

The [paper's §3.3](https://arxiv.org/html/2407.16394v1#S3.SS3) describes supervised
matching of corresponding pose/RGB clips, intended to align clip–word associations
indirectly. That matches the broad purpose of the enabled branch, rather than
requiring the optional top-k KL. Read scope: metadata, abstract, method sections
3.1–3.3; no full-paper or published-result reproduction. Exact equation/source
parity is **not** certified: the rendered equations and the single-softmax source
need separate scrutiny if that question becomes decision-relevant.

## Executed source checks and honest failure trace

Only AST-extracted argument declarations/get_args and one constructor assignment
were executed. No upstream imports, shell launcher, process group, model,
checkpoint, features, annotation files or evaluation were executed/accessed.
Scripts were shell-tokenized, not shell-executed. Recorded hashes cover parser,
model source, three launchers and audit script.

First audit-script attempt incorrectly assumed freeze_exfusion was a declared
flag; its assertion failed. Inspection found no declaration or assignment to
the args attribute in the searched source. The second attempt executed full
recipe parsing and failed on fusion_type under Python3.13.5. The final audit
records both findings instead of assuming a default or silently repairing source.
It exits0 because it successfully records expected limitations; this is **not**
a successful SEDS startup/training result. This was exploratory source checking,
not a preregistered efficacy experiment.

Two distinct reproduction findings:

1. `task_config.freeze_exfusion` is read unconditionally in the constructor.
   The full extracted parser with `--do_train` does not produce this attribute;
   executing the exact assignment raises AttributeError. Main passes args to
   model creation; inspected logger/device initialization does not supply it.
   Full startup was not run; earlier dependency/resource failures could intervene.
2. `fusion_type` choices is a single string rather than a collection of option
   strings. Full parsing of each launcher fails in current Python3.13.5, whose
   argparse checks individual characters. This must **not** be called an original
   Python3.10 failure: the README recommends3.10, and official
   [CPython3.10.14 source](https://raw.githubusercontent.com/python/cpython/v3.10.14/Lib/argparse.py)
   uses direct membership in the choices object, allowing substring membership
   for strings. The implementation was verified in the official raw source
   `_check_value`; no Python3.10 runtime was executed. The six loss flags are
   independently parsed using their unmodified AST declarations, which excludes
   this unrelated fusion-option check. This targeted parse is not full CLI parity.

## Boundaries and next work

The previous repo summary accurately described existing KL code but lacked this
activation qualification; it is corrected now. Reproduction issues are not novel
research contributions and do not disprove the paper's empirical findings.
No default freeze_exfusion value was invented, no upstream repair was applied,
and no forbidden SEDS checkpoint/features/keypoints were requested.

Source and executable checks support narrow configuration claims; causal model
harm remains unmeasured. No systematic prior-art claim, independent human review,
DOI-resolution audit or full statistical study is claimed. Public paper material
is methodological context, not a benchmark-selection signal.

Next SEDS analysis, if any, must concern an actually enabled mechanism and remain
source-only under the asset prohibition. Generic pose fusion, distillation,
support mining and alignment changes remain subject to the binding closures.
Do not retry optional-KL diagnostics or build a replacement SEDS training pipeline
merely to work around this gate. Overall search remains open; no Proposal8/Q38/GO
or global blocker follows from this configuration correction.
