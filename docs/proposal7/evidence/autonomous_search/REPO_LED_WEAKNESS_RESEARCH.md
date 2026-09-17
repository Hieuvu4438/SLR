# Repository-led weakness research — active direction

## Authority and current conclusion

2026-09-15. The user explicitly requested research into CMCM, SAN, SEDS, SLRT,
UPRet (assuming “UPRER” refers to UPRet), CiCo and related implementations to find
weaknesses and develop improvements toward SOTA. This adds emphasis to the active
autonomous goal, not permission for TEST selection, closed-family relaunches or
SEDS asset use. Skill: academic-research-suite, source fact-check inline.
ANALYZED, AI-assisted; no independent human/linguistic review.

Current cross-repository synthesis (2026-09-16):
[repository-gap overview](REPOSITORY_GAP_OVERVIEW.md). It covers confirmed defects,
unresolved empirical consequences, contrary evidence, task/resource limits and
method-discovery prerequisites. The original transport-reduction prioritization
below is historical, superseded by its completed negative sensitivity screen.
Fixing a reproduction discrepancy alone remains a baseline control, not a research
contribution or SOTA result. No method has been selected.

Latest outcome: [real-TRAIN gradient screen](UPRET_real_gradient_result.md)
completed and exactly replayed. The frozen-amplitude-controlled median gradient
change is .174891% visual/.220684% text at the partial checkpoint, below the
locked1% sensitivity threshold. Deprioritize a long reduction campaign; this is
not evidence of universal ineffectiveness. The historical next-experiment gates
below have been superseded by that measured decision. No method GO.

## Pinned source comparison

New [UPRet deterministic-loss bound](UPRET_loss_rank_bound_result.md),2026-09-16:
default auxiliary-score margin perturbation at most .75 under stated finite/
unit-mass assumptions. Saved no-OT losses imply strict deterministic-channel
separation on all three existing B32 partial-step TRAIN batch states. Posthoc
arithmetic inference, not fresh rank evaluation/full-gallery efficacy. Reject
auxiliary-only batch success; do not reopen weight/reduction/temperature sweeps.
This interpretation does not transfer to the completed strong CiCo baseline.

New [CMCM Gaussian loss contract](CMCM_gaussian_result.md),2026-09-16:
asymmetric epsilon placement gives negative loss for matched Gaussians and an
unbounded common-variance-decreasing parameter path. Six unchanged-source CPU
fixtures validate formula and derivative. Not observed trained collapse, full
objective unboundedness or retrieval harm. Ordinary KL correction is not novelty;
no matched CMCM integration/activation resource established or replacement built.

New [CiCo visual-mask information flow](CICO_visual_mask_result.md),2026-09-16:
source contract and fixed fixture rule out padded-input contamination of valid
states in 2D/sum path. Masked positions still read valid content and enter one
scorer channel. No trained usefulness or C14-effect attribution follows. Learned
query pooling is prior art; no mask/register/query-count rescue justified.

New [CiCo distributed-gradient contract](CICO_distributed_gradient_result.md),
2026-09-16: source backward with replicated full loss and emulated DDP average
scales encoder gradients1/W but not temperature. Nine fixed source-AST controls
pass to1.39e-17. The historical local baseline path does not enable this gather,
so no current-error attribution or corrective training follows. Not actual DDP
runtime, an Adam update effect or a novel method. No optimizer/temperature rescue.

Follow-up [C²RL retrieval task/resource gate](C2RL_retrieval_resource_gate.md),
2026-09-16: the independent release's evaluators are SLT generation/BLEU and batch
VLP losses; no compatible trained retrieval resource verified in its sole branch,
empty tags/releases, full README and untruncated tree. mBART initialization is not
a trained retrieval checkpoint. Documented --eval selects TEST and was not run.
Two C²RL turns complete; retain kernel finding without retrieval-harm inference,
then move mechanism. No repeated lookup, masking rescue or substitute pipeline.

New [C²RL-derived reproduction source](C2RL_reimplementation_source_result.md),
2026-09-16: independent sltbaselines partial-author-code attribution verified as
a repository statement. Enabled kernel's padded-width dependence reproduces in
a fixed synthetic I2T-order reversal. Not original-C²RL retrieval parity, trained
harm or novel masking. No output assets or replacement pipeline acquired.

New [CiCo visual radial feasibility](CICO_visual_radial_result.md),2026-09-16:
both fixed magnitude endpoints feasible on155/256TRAIN directions at the final
LayerNorm/projection only. Row0 construction prerequisite failed and was not
replaced. No actual-video collision, semantic harm or training justification.
Two radial turns now complete; no scale/precision/row rescue.

New [CiCo radial-information check](CICO_radial_contract_result.md),2026-09-16:
LayerNorm permits an exact text-radius inverse in ideal arithmetic, but native
TRAIN-cache p95 error11.770844% fails the locked1% screen. Ill-conditioned learned
projection and a separate FP16 synthetic fixture preclude interpreting failure as
semantic information loss. No cosine-removal, norm gate or precision rescue
justified; no retrieval effect measured. This is a Q01 claim-boundary refinement.

New [SEDS temporal-input contract](SEDS_temporal_contract_result.md),2026-09-16:
enabled matching assumes corresponding pose/RGB clip windows; loader checks counts,
not source-frame provenance. Paper describes shared preparation, so misalignment
is NOT established. RGB loader and pose-window functions are AST-identical across
three TRAIN loaders. No SEDS assets/producer download or empirical harm claim.
This does not justify an alignment-repair campaign; move to a different mechanism.

New [SEDS activation gate](SEDS_loss_activation_result.md),2026-09-16:
top3/top5KL is optional/defaultFalse and omitted in all3 supplied training scripts.
Enabled auxiliary branch is clip-index Pose–RGB matching with coefficient.4.
Earlier KL source inspection must not be attributed to default released training.
Missing freeze_exfusion setting and currentPython3.13-specific fusion CLI failure
are reproduction issues, not methods or evidence of poor published performance.
No SEDS assets, model execution or upstream repair.

New [SAN TRAIN exposure check](SAN_train_collision_result.md),2026-09-16:
actual German7096TRAIN captions have zero distinct-caption single-swap collisions,
independently verified over769199transpositions. Max identical multiplicity63
excludes all-identical full B64 batches in the supplied no-replacement setup.
Textaugment source semantics verified conditionally, author dependency unpinned.
Both public branches/releases checked; no actual negative table found there.
No repair training justified, no global claim about SAN effectiveness. Move on
from this two-turn source/exposure branch rather than repeat it.

New [SAN negative-contract check](SAN_negative_contract_result.md),2026-09-16:
eight synthetic source fixtures verify missing duplicate/self-output guards,
conditional empty-pool failure, post-generation mocked-swap collision and
multiplicity weighting. Ownership indexing passes. Real table/EDA/tokenizer
exposure is unknown; no semantic-label or retrieval-harm claim. These are
reproduction prerequisites, not a novel method or permission to reopen mining.

New [CiCo channel-objective check](CICO_channel_objective_result.md),2026-09-16:
separate-channel CE differs from CE of mean logits by a known nonnegative
agreement term, verified against actual CrossEn and derivatives in four fixtures.
Augmented training inputs and coupled-channel realizability prevent an automatic
deployment-harm interpretation. Primary ensemble-decomposition/collusion prior
blocks generic mixed-loss/interpolation novelty. No training or new candidate.

Subsequent [CMCM source audit](CMCM_source_result.md) found and isolated a
covariance forward/backward epsilon discrepancy on valid PSD feature inputs.
Large errors occur at low trace; trained exposure is unknown. This is a verified
numerical prerequisite, not a demonstrated retrieval bottleneck or method GO.

The [trained-resource gate](CMCM_resource_gate.md),2026-09-16, now verifies BOTH
public branches and empty tags/releases: alternate main is README-only; master
matches the pinned module package. Neither supplies the trained pre-covariance
state/integration needed for exposure attribution. No generic-weight substitution
or replacement pipeline is justified; this is a specific resource dependency,
not a global research barrier or criticism of published performance.

Latest [CiCo pseudo-clip census](CICO_pseudoclip_result.md) measures exposure of
the already-documented first-anchor merge in the existing local P14T adaptation
index. Only68/9854 training segments have gaps; expected any-gap crop probability
is0.615417%, below the locked10% gate. All-row frame/interval checks and sampler
AST checks pass; replay exact. No actual label-error, retrieval-harm or author-
released-feature claim. Deprioritize this repair route, keep OCEM closed, and
change research question before another efficacy probe.

The subsequent [subword identity check](SUBWORD_resource_and_collision_audit.md)
does not establish a CiCo spelling-loss bottleneck. Its tokenizer retains inverse
byte/symbol mappings; trained accessibility is a separate question. How2Sign's
published FS gloss convention has no verified current downloadable label join,
while FSS-Net and character-level sentence translation are prior art. No new
character head or pseudo-label/extra-stream method is promoted. The linked note
also discloses and quarantines unsolicited TEST examples in a public issue;
no TEST file or evaluation was accessed.

[SignRep scope/feasibility check](SIGNREP_scope_and_feasibility.md),2026-09-16:
new official source inspected at06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2.
Different pretraining and dictionary-retrieval task; no matched sentence-retrieval
effect inferred. Feature/latent interface checked, no assets or execution.
AS-C26/C27 and AS-C42 reread to avoid repeating inadequate or already-controlled
probes. That handoff is now completed in the
[all37 evidence/feasibility audit](Q01_Q37_feasibility_audit.md). It preserves
unresolved broader questions instead of promoting narrow negatives into global
exhaustion. The [Q02/Q22 analytical follow-up](Q02_Q22_function_class_boundary.md)
also rejects shared-score/dimension-count arguments as sufficient evidence of a
retrieval ceiling; actual scorer restrictions and learnability remain unresolved.
No new run, novel candidate, barrier or GO follows. Next is Q01 positive-control
adequacy analysis without another residual-model recipe rescue.

| Repository | Pinned local revision | Inspected scope / decision |
|---|---|---|
| UPRet | 046366227417e1d8ec14145965403462df345984 | Committed `modeling.py` Sinkhorn and probability-score path; working `PDE.py`. Primary target for controlled reduction comparison. Existing working-tree mask/data/training patches are NOT author code and were preserved. |
| SAN | 82aba9cbc1beb403abef6e9a3875ca52479805c8 | `models.py` gathering, hard-negative scoring and `train_vlp_v2.py` evaluation. Previously recorded TEST-loader selection and inner-padding issues are reproduction controls, not new discoveries. Train uses exp(logit_scale), eval raw logit_scale: positive scaling alone preserves ranks, so not a promised recall improvement. |
| CMCM | 5d458719d1da2f082e188cc44705003d919e7e97 | Full TMCP/CSA/Encoder and CCG call paths inspected. Undefined DEVICE, mask slicing and encoder-shape concerns already documented; public package lacks a verified runnable retrieval driver. Do not mistake repair of these issues for evidence against the published method. |
| SEDS | 434e3f714fcb6a7d1f4001fb9a246bbd93ec0246 | Top-k KL code inspected but DISABLED in all3 supplied training recipes; enabled auxiliary loss is clip-index Pose–RGB matching. Source/CLI AST only, no assets. See activation gate for reproduction limitations; generic distillation/support changes remain closed. |
| SLRT / CiCo | 38a4f7b00da7a858d59b7fabe5093876a84db8e0 | Existing verified baseline and earlier score/encoder/training audits are the control foundation. No new full-tree audit claimed in this pass. Earlier failed padding/temperature/ensemble variants will not be rerun as new ideas. |

Official repository pages were checked for identity:
[UPRet](https://github.com/xua222/UPRet), [SAN](https://github.com/joonmy/SAN),
[CMCM](https://github.com/vddong-zjut/CMCM), [SEDS](https://github.com/longtaojiang/SEDS).
No pull/reset/commit/push or upstream file edits were performed. Pinned revisions,
not claims about an unseen latest commit, define the inspected code.

## UPRet: distinguish intended approximation from an implementation discrepancy

The original [UPRet paper](https://arxiv.org/pdf/2405.19689), §§3.3–3.6, explicitly
fixes the plan while optimizing and uses its OT term only during training.
Neither detached gradients nor removal of that term at inference is therefore
an accidental omission. Equation21 describes a transport-weighted sum. The
inspected source instead pools each random sample over its tokens, forms a2×2
sample-to-sample matrix, multiplies by the plan, and averages row/column maxima.
The support concern is resolved by the final-paper check below; only the reduction
discrepancy remains. Original read scope: title/abstract and §§3.1–3.6, arXivv1.
No published performance was reproduced or used for selection.

New executable certificate:
[protocol](UPRET_gradient_protocol.md), [run](UPRET-GRADIENT_run.json).
Actual committed AST for Sinkhorn and transport block executed in isolation;
no upstream module imports, dataset, checkpoint or pretrained assets.32 fixed
CPU FP64 matrices in[−.5,.5], each explicitly realized by normalized R4 vectors.
Parameters match source: sample_num2, eps.1, max_iter100. First run exit0,.737017s.

- All32 backward gradients differ from numerical full-forward derivatives by
  >1e−5; largest coordinate difference .184077. This establishes a surrogate
  derivative, not harmful optimization. The paper explicitly motivates plan fixing.
- Finite differences holding the plan fixed agree with autograd within2.22e−11.
-31/32 cases have finite-difference step discrepancy<1e−7. The remaining case
  has discrepancy1.39308e−4, so blanket numerical smoothness is NOT claimed.
- An independent closed-form uniform2×2 plan also produces full-versus-frozen
  derivative differences in all32 cases, largest .184078. This is not solely a
  Sinkhorn early-stop artifact; maximum source-versus-exact plan discrepancy is
  .000723841. These are not parameter-gradient or benchmark measurements.

For similarity matrix S and eps=.1, the exact uniform plan is
T=[[a,.5−a],[.5−a,a]], a=.5 sigmoid((S00+S11−S01−S10)/(.2)).
Unit tests check marginals, kernel cross-ratio, and analytic-gradient agreement
with finite differences. This derivation is a local mathematical control, not
an asserted novel transport result.

Posthoc comparison, explicitly not preregistered as part of the original32-case
gradient test: [UPRET-REDUCTION_run.json](UPRET-REDUCTION_run.json).
Same bank, no added/selected cases. Independently recomputed source reductions
match saved values within1e−14. Replacing the max reduction with a full
transport-weighted sum reverses **59/496** strict pairwise score orders; the
exact-plan variant gives the same59 reversals. This is not a constant rescaling,
but also not a retrieval improvement: none of these synthetic pairs has a
semantic relevance label or a corresponding PH query. Both runs exited0.

Full probe regression suite after both new analytical fixtures:83passed1.82s,
exit0. This validates helper behavior, not a retrieval efficacy claim.

## Next experiment gate

### Final-paper and baseline-readiness update, 2026-09-15

Origin Skill: academic-research-suite / source verification (inline).
Verification Status: ANALYZED; no independent human review.

The [official ECCV version](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06074.pdf)
§3.3 explicitly pools samples over the token dimension. §3.5 defines K-sample
uniform marginals, and Eq18 is the full transport-weighted similarity sum.
Thus sample-level support is intentional, while the source max reduction still
differs. §4.3 confirms the added modules are inactive at inference. The explicit
fixed-plan explanation cited above belongs to arXivv1; it is not present in the
inspected final methodology. Do not misattribute that wording to the final paper.
Read scope: final §§3.1–3.6 and §4.3 efficiency paragraph, browser PDF text.

Existing integration: `methods/sssc/method1/baseline.py:292` implements the
two-sample max reduction; `baseline_loss_from_local` adds it outside the CLIP
logit-scale multiplication. Preserve that placement during a reduction-only
comparison. Existing masking, augmentation and RNG repairs are documented in
`methods/sssc/docs/baseline_repairs.md`, not new contributions from this audit.
Do not edit this package in place: its source identity gates historical resumes.

Read-only executable [readiness census](UPRET-READINESS_run.json) completed
exit0, .177814s. All49 recorded source files match the historical run manifest.
A separate call to the existing full-tree provenance implementation confirms
the complete semantic source hash still matches, including its enumerated files.
No checkpoint, data or TEST content was loaded. No training was launched.

There are82 saved DEV evaluations, through step1066 of a declared2600 updates.
The separate latest-train JSON records1060; neither is a checkpoint-resume
certificate. No training-complete marker exists. Highest saved mean bidirectional
R1 is19.845857 at step767; last saved mean is18.497110. These are historical
local-resource run records, not new evaluations or published-baseline metrics.
The old README/runbook status is stale relative to these files. Feature provenance
also differs from the existing strong CiCo run and from the published release;
see the existing Method1 resource ledger. Reading that historical ledger does not
mean its old TEST audit was repeated here. Broad SOTA comparability is not met.

Decision: the mathematical reduction lead survives, but a trained, competitive,
resource-matched UPRet control is NOT yet established. Before launching three
expensive arms, register a real-TRAIN-batch sensitivity check with exact shared
encodings/noise and the actual combined loss, comparing max, sum and a frozen
TRAIN-calibrated amplitude control. A changed synthetic ordering alone cannot
show appreciable encoder-gradient influence. Preserve detached plans, temperature,
feature regime and all other factors. Do not reuse the incomplete checkpoint as
a completed baseline or silently equate Method1 features with current CiCo inputs.
This prerequisite does not reopen SSSC or the closed optimizer/initialization sweeps.

Validation: combined Method1 baseline fixtures and information-probe suite,
88 passed in1.92s, exit0. This is helper/integration regression coverage, not a
reproduction of the saved training curve or an efficacy result. Our probe worker
is terminal; an unrelated GPU process was observed and left untouched.

Prioritize a UPRet-native, matched-input control before method invention:

1. Reconcile the final paper/code version and establish a runnable DEV-only
   UPRet baseline using permitted, provenance-matched assets. Inspect the existing
   local baseline integration and prior logs before launching anything.
2. Pre-register the *single reduction change* (max versus sum) with identical
   initialization, sample tensors/RNG, stop-gradient plan, objective weights,
   updates and exposure. Keep the original as B0 and a scale-matched sum control
   to separate score amplitude from order effects. No simultaneous gradient fix.
3. Measure training gradients and true deployed DEV ranking—not merely a lower
   auxiliary loss. An intentional surrogate may outperform the exact derivative;
   do not assume differentiation is the improvement.
4. Only an attributable deployment gain plus a distinct, internally open causal
   mechanism can motivate three candidates and the required novelty/adversarial
   gates. Generic OT, uncertainty or teacher-preservation variants remain closed.

No benchmark training has been launched from this finding. Baseline fidelity is
the next necessary gate, not a claim that UPRet already beats the existing strong
three-model control. SAN and CMCM remain alternative source-led investigations;
their missing-resource/runtime issues do not justify invented reproductions.

## Audit limitations

Partial repository coverage, no SOTA evidence, no author-conduct allegation, no
independent experimental replication. Public code is authoritative for its own
operations, not for undisclosed author runs. Paper and repository share authors,
so they are not independent evidence of efficacy. Primary computational paper
(LevelIII design broadly; descriptive equation claims) and source documentation
(LevelVI) have high fitness for comparison, not for causal performance claims.
Bibliometric/retraction databases were not checked; no such certification claimed.
The academic-research-suite source checks prevented promoting the intended
detachment/inference behavior as a bug and narrowed the next control to reduction.
