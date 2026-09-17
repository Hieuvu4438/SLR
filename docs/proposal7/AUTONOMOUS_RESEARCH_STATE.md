# Autonomous SLRet research state

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-14
- Verification Status: ANALYZED (R0 independently replayed; pilots are not a verified method)
- Version Label: autonomous-v2

## Governing scope

The user-selected `Astra_SLRet_Autonomous_Research_Loop.md` supersedes the
earlier Phase 2 three-cycle stopping rule. The information matrix is ONE branch,
not the entire search. Preserve all closed families and the unchanged Proposal 7
numerical gates. No test access, SEDS assets, changed positives, or new benchmark.
After two failed cycles in a causal layer, move to a different layer.
Success requires novelty, controlled empirical survival, and the conditional
second-dataset check—not merely an interesting diagnostic or a passing test.

## Current status: first selectively reopened CiCo pilot completed, failed lead gate (2026-09-16)

Read [CICO-REOPEN-01 result](evidence/autonomous_search/CICO-REOPEN-01_result.md).
Actual three-head training completed780updates total, exit0; all9saved outputs
replay exactly without training. Sentence head final meanR1=74.759152 versus
75.240848CiCo; blind74.855491, shuffled74.566474. All selectors retain step0.
Sentence final persistent mean ranks worsen.673913/.494253; some R5/R10 improve,
but the locked lead fails. Head parameters/weights changed, so this was not an
inactive-intervention test. No larger weighting campaign or tuning rescue.

Seven focused tests pass. Recorded runtime6.922318s, peak1816695296GPUbytes;
head-only cached computation, not end-to-end speed. Validation exit0,2.974778s.
All run/validation/protocol hashes and checkpoint/score/metric artifacts retained.
No TEST/SEDS assets or upstream edits; no worker active. Goal active, no GO.
Next move to an explicitly justified encoder-side reopening, not more pooling;
first inspect historical code/controls before naming or training that mechanism.

### Selective reopening authority and completed pilot setup

The user answered: "Allow selective reopening with explicit justification and
controls." This supersedes the prior unanswered-question status below. Preserve
all negative measurements and all unreopened closures. No SEDS-asset or TEST
permission changed. Registry and focused build plan now record the amendment.

First exception is [CICO-REOPEN-01 protocol](evidence/autonomous_search/CICO-REOPEN-01_protocol.md):
learned sentence-conditioned outer clip weighting, compared with unchanged CiCo,
query-independent and random TRAIN-conditioning heads. The old inference-only
inner-pooling results are not rerun. Prior-inspired engineering screen, not novel
method or GO. Three heads32800params each, frozen CiCo caches, common B64 schedule,
seed42,260updates/head, no augmentation, evaluations0/130/260,15min hard timeout.

Code in methods/information_probe/sentence_weighting.py and sentence_weight_pilot.py.
Initial unit test found9.54e-7 rounding mismatch from multiply/divide order; source
arithmetic reordered to match existing scorer without relaxing exact test tolerance.
No training launched before those tests. Intended outputs are the fixed pilot
artifact directory and CICO-REOPEN-01_run.json; do not overwrite/retry a failed run.

### Previous CiCo focus handoff (scope question now resolved)

The latest user instruction prioritizes building on CiCo to surpass CiCo, SEDS
and comparable SLRet work. Read [CICO_FOCUSED_BUILD_PLAN.md](CICO_FOCUSED_BUILD_PLAN.md).
Broad repo overviews are no longer the primary deliverable. Retain existing
baseline/evaluator interfaces and controls; no new generic training framework.

CiCo/SEDS scoring and SEDS fusion call paths rechecked. SEDS adds representation/
fusion and cross-stream objectives but retains CiCo-form token interactions.
CiCo already admits a query-token-conditioned pooling interpretation. Primary
X-Pool §§4.2–4.4 provide a concrete prior-art boundary for a generic query-aware
attention adaptation, not a new measured failure or a candidate GO. No pooling
rescue justified; C24/C25 and prior method closures remain binding.

Optional user question pending: allow selective reopening of inconclusively
tested conceptual families, or preserve blanket closures? The new build focus
is not treated as an implicit answer. No training/model/data/TEST/SEDS assets
accessed, no upstream changes. This scope question is not a global barrier.
Next deliverable is an eligible minimal method specification and controlled
pilot implementation, not another overview. No GO/Proposal8; goal active.

### Completed repository overview and UPRet batch explanation check

The user's latest instruction explicitly prioritizes a comprehensive repository-gap
overview before method selection. Read
[REPOSITORY_GAP_OVERVIEW.md](evidence/autonomous_search/REPOSITORY_GAP_OVERVIEW.md).
It consolidates all requested names, pins, confirmed defects, contrary evidence,
unresolved consequences, source/task coverage limits and advancement requirements.
SLRT contains CiCo; they are not independent sentence-retrieval baselines. This
is a bounded synthesis checkpoint, not a fresh whole-repository audit, current-SOTA
certification, new empirical cycle for every row, or global exhaustion table.

New evidence: [UPRet loss/rank bound](evidence/autonomous_search/UPRET_loss_rank_bound_result.md).
Source trains deterministic logits jointly with additive transport. The default
source auxiliary term changes any margin by at most .75 under finite/unit-mass
conditions. Posthoc saved no-OT losses imply every directional-channel B32 pair
is correctly separated on all three partial-step TRAIN batch states, with minimum
derived bounds4.516722/3.199878/5.177416. No new model/rank evaluation; FP32 loss
inference is not an interval-certified result. This rejects auxiliary-only batch
success, not full-gallery or semantic failure. Existing gradient-screen negative
and incomplete/resource-mismatched UPRet baseline limits remain unchanged.

Artifacts: calculation script, UPRET-LOSS-RANK-BOUND.json, result note, overview
and synthesis claim-intent passport. Only saved JSON/config/source were read by
the calculation, exit0. No training, GPU, DEV/TEST content or SEDS assets. Existing
upstream edits preserved. No worker active; goal active; no method GO/Proposal8.

Next: move from this completed auxiliary-objective check to a distinct unresolved
representation/scorer-interface formulation from the overview. Require a specific
refutable consequence and an admissible intervention before new code/training.
Do not run another overview-only cycle, revive inadequate residual heads, or
promote generic negative mining/distillation from the train/generalization gap.
The older source-corrected CiCo training question remains unanswered; no such
campaign was launched or treated as authorized by automatic continuation.

### Completed alignment-resource validation-use gate

Read evidence/autonomous_search/PH_alignment_provenance_gate.md. Officialdirectory
lists PH2014T alignment resources; primaryPAMI methods and pinnedRe-Sign source
verify an iterative weak-label/model-based timing pipeline. No archive inspected
or joined. Such labels cannot independently validate actual cues merely through
probe agreement; linguistic labels are inputs to their construction. No claim
that all automatic labels are wrong or unusable, or that archive parity is verified.

The preceding source-correction scope question was a no-progress turn, not new
authority or a global blocker. No corrected-baseline training launched. Two
annotation-resource passes complete: move causal layer, no more archive/teacher/
probe rescue. No dataset/TEST examples/assets/GPU/training. Goalactive,no worker;
no Q38/Proposal8/GO/globalbarrier.

### Completed complex TRAIN notation scope

Read evidence/autonomous_search/PH_complex_notation_result.md. Existing7,096-row
complex CSV retains __EMP__285rows,__LEFTHAND__312,__HOLD__610; standard andcomplex
both have neg-175tokens/132rows. Official codebook defines head-shake negation;
do not claim all nonmanual annotation signals absent. However detailedmb:/mk:/lh:
andotherpostfix fields, parentheses and variant# are absent; timestamps−1.
Literal counts are not cue-isolating anatomy/timing, completeness or retrieval
labels. Prior schema-only audit did not interpret these tags; new scope is now
recorded. No fuzzy-gloss/polarity mining, generic probe or gloss-training campaign.

No model/features/video/DEV/TEST/GPU/training. Existingresource can support
descriptive annotation work but does not resolveQ17's causalcue isolation orQ01's
adequacy. NoQ38/Proposal8/GO/globalbarrier. Goalactive,no workeractive.

### Completed CMCM Gaussian-loss contract check

Read evidence/autonomous_search/CMCM_gaussian_result.md. With identical means and
variances, released GaussianAlignmentModule has negative loss and an unbounded
decreasing common-variance path in real arithmetic. Six unchanged-source FP64
CPUfixtures agree with formula/gradient to1.43e−14/1.67e−16. Source remains
unmodified. This is separate from the earlier covariance backward discrepancy.

Actual trained exposure/integration remains unavailable; no collapse or retrieval
harm measured. Ordinary consistent-KL/variance-floor/deletion repairs are not
novel candidates. No trained weights/data/DEV/TEST/GPU/optimizer steps. Move away
from numerical repair; no repeated resource search or replacement pipeline.
No Q38/Proposal8/GO/global barrier. Goal active, no worker active.

### Completed raw-input short-padding exposure check

Read evidence/autonomous_search/CICO_input_padding_result.md. All7,096TRAIN
metadata files: decoded length16–475,0below16,13equal16; all720,914windows exactly
match16-frame stride-one geometry. No repeat-last padding is required by short
inputs, and no final-window shift is needed. Source/metadata hashes recorded;
read-only census exit0. This does not audit decode failures or internal I3D
convolution padding, establish retrieval harm, or describe pretraining inputs.

No model/features/checkpoints/video/DEV/TEST/GPU/training. Do not run a padding
repair or reinterpret the earlier pseudo-clip census. Move to a distinct open
mechanism; no Q38/Proposal8/GO/global barrier. Goal active, no worker active.

### Completed human TRAIN audit semantic scope

Read evidence/autonomous_search/SACREBIRD_semantic_scope_result.md. All307comments
read via150distinct strings with complete row memberships. Primary §3 confirms
gloss/German comparison and explicitly excludes reordering labels. Fixed first
comment: official glossFEBRUAR versus Germanaugust and EnglishAugust. Disagreement
verified, signed month unobserved; this cannot identify a visual–English error.
No new per-example labels, clean subset, model-score correlations or repair.

Resource remains useful descriptive evidence but does not resolve Q06/Q09 order
controls, Q11 relevance, Q17 cue isolation, Q23 irreducibility or Q31 context joins.
Two annotation passes complete: move causal layer, no more comment-category/row
expansion or TEST-backtranslation access to rescue this use. No model/feature/
checkpoint/GPU/training/DEV scoring/TEST CSV access. No Q38/Proposal8/GO/global
barrier. Goal active, no worker active.

### Completed existing human TRAIN audit join

Read evidence/autonomous_search/SACREBIRD_train_join_result.md. New DFKI
sacre-bird-phoenix pin012b11c supplies307human gloss/German-text consistency
annotations:307exactTRAIN matches,0DEV/nonTRAIN/duplicate IDs. Raw category1
counts97gloss-omission/13German-omission/33lexical/165minor, overlapping categories,
not population rates or signed-video relevance labels. Exit0, source/code hashes
recorded. Only allowlisted TRAIN CSV fetched; comments not semantically inspected.

Primary §4.1 verifies structured consecutive-block sampling and one deaf fluent
L2DGS annotator. No multiple-rater or random-sample assumption. This changes the
TRAIN consistency resource gate, not Q11/Q17/Q23 or DEV-ambiguity evidence.
Next: bounded TRAIN comment/codebook scope check before any model diagnostic;
do not relabel/filter examples or reopen noise/soft-positive/lexical closures.
No TEST CSV, model scores/assets/checkpoints/GPU/training. No Q38/Proposal8/GO
or global barrier. Goal active, no worker active.

### Completed CiCo visual-mask information flow

Read evidence/autonomous_search/CICO_visual_mask_result.md. Configured 2D/sum
encoder masks padding and CLS as keys, not queries: valid states cannot read
masked inputs, but masked states can read valid content. Source-AST fixed CPU
fixture gives zero valid-output change/pad-input derivative; masked outputs
change .341427028 and have valid-input gradient norm .705776989. Random two-layer
model only, not trained semantic/gain evidence. Exit0, certificate hash verified.

This rules out the specific encoder-contamination claim and clarifies that C14
removed potentially content-dependent readouts, not only zeros. It does not
explain C14's negative performance effect. Learned-query pooling has prior art;
register analogy does not establish defect/novelty. No mask/query/register sweep,
data/checkpoint/GPU/TEST access or upstream change. Move mechanism; no Q38,
Proposal8, GO or global barrier. Goal active, no worker active.

### Completed CiCo distributed-gradient contract

Read evidence/autonomous_search/CICO_distributed_gradient_result.md. Pinned
AllGather backward plus emulated DDP averaging yields encoder1/W and temperature1
gradient factors under identical global inputs. Nine fixed CPU source-AST cases
pass; correction controls distinguish feature-gradient from whole-loss scaling.
Max coordinate error1.39e-17. Transport/DDP emulated, no multiprocess/GPU claim.

Recorded local baseline uses single-device direct construction with distributed
None; historical source path verified after resolving old elsc/ layout. Thus this
is not an active local continuation error mechanism. Unknown release pretraining
effects remain unknown; Adam/clipping prevents learning-rate-effect inference.
No training/data/checkpoint/TEST access, upstream change, optimization sweep or
novel candidate justified. Move mechanism, not more world-size fixtures.
No Q38/Proposal8/GO/global barrier. Goal active, no worker active.

### Completed C²RL retrieval task/resource gate

Read evidence/autonomous_search/C2RL_retrieval_resource_gate.md. Independent
sltbaselines supplies SLT generation/BLEU evaluation and batch VLP loss evaluation,
not a verified full-gallery C²RL retrieval pipeline. One public branch at the
same pin, no tags/releases, untruncated tree; no compatible trained retrieval
checkpoint identified in those checked locations. Linked mBART resources are
initialization, not verified C²RL retrieval weights. No global absence claim.

Documented SLT --eval selects TEST; do not execute under DEV-only constraints.
Only code/config/README/API metadata read, no assets, TEST contents or model runs.
Prior padding fixture remains kernel evidence, not trained retrieval harm.
Two C²RL turns complete: move to a different mechanism; no masking rescue,
replacement SLT pipeline or repeated resource lookup. No candidate/Q38/Proposal8,
GO or global barrier. Goal active, no worker active.

### Completed C²RL-derived reproduction kernel audit

Read evidence/autonomous_search/C2RL_reimplementation_source_result.md. Independent
sltbaselines commitf741330d contains a kernel attributed to partially shared C²RL
author code; NOT verified original retrieval release or published-recipe parity.
Enabled c2rl training adds its loss to translation loss. Dynamic padding remains
in inner softmax after1e-5attenuation. Fixed source-AST CPU fixture changes I2T
pair ordering at widths2→32 with identical valid features; independent formula
matches<=2.78e-17, T2I/control unchanged. No mixed-score/semantic/gain claim.

Only source/config/README fetched, no outputs predictions, datasets, checkpoints,
GPU, full encoder/training, TEST examples or SEDS assets. Exact4cuda-call CPU
adaptation disclosed. This is a reproduction concern, not a novel masking method.
Original-C²RL retrieval resource gap remains. No replacement SLT pipeline or
masking rescue authorized. No candidate/Q38/Proposal8/GO/global barrier.
Goalactive, no worker active.

### Completed visual radial feasibility check

Read evidence/autonomous_search/CICO_visual_radial_result.md. Actual visual final
LayerNorm/projection admits both .9 and1.1 times stored magnitude for155/256fixed
TRAIN directions; .9 feasible256/256,1.1 feasible155/256. Constraint matrix rank
513/513, residual<=4.30e-14. This is local function-class feasibility, NOT actual
video collisions, end-to-end reachability or useful semantic information loss.

Preregistered row0 construction infeasible at one endpoint and skipped; no
favorable-row substitution, no native/FP64 row0 forward-parity claim. Two new
synthetic algebra tests plus3text tests pass. Existing text certificate source
and parent hashes verified. No encoder/scorer, training, GPU, DEV/TEST cache or
SEDS asset access. Text-omission and score-derivative audits already completed,
so not rerun. Both radial turns complete: move mechanism, no scale/precision
rescue. No Q38/Proposal8/GO or global barrier. Goalactive, no worker active.

### Completed text radial-information check

Read evidence/autonomous_search/CICO_radial_contract_result.md. Post-LayerNorm
text norm has an exact affine-constraint inverse under stated conditions, but
native TRAIN cache reconstruction fails the locked1% p95 screen:11.770844%
on125246valid tokens across7096rows. Learned projection condition3824.98.
Exact synthetic control passes; posthoc source-LayerNorm/FP16-projection CPU
fixture produces2.072366% p95 error. This shows precision can break the inverse,
not that it explains all real-cache error or that lost magnitude is semantic.

Original certificate and separate posthoc validation both exit0;3unit tests pass.
No encoder, training, GPU, retrieval scoring, DEV/TEST cache or SEDS assets used.
One historically DEV-selected checkpoint inspected, no new selection. Visual
projection rectangular, outside this inverse; no visual-loss inference.
No norm/confidence/precision rescue or new method justified. Q01 boundary refined;
no Q38/Proposal8/GO or global barrier. Goal active, no worker active.

### Completed newer-literature transfer screen

Read evidence/autonomous_search/SignMatch_transfer_scope_screen.md. Newly located
SignMatch uses temporally annotated sign classes and dictionary alignment for
visual-form matching. This is an adjacent prior, not a matched sentence-retrieval
baseline. Official project page links no implementation yet; no global code-
absence claim. Selected manuscript sections read, not full paper/code audited.

Direct import changes resources/supervision; teacher-mining adaptations collide
with closed families; unlabeled prototypes lack a measured bottleneck/mechanism.
No prototype training campaign justified. WSLP spotting exclusion only rechecked;
SignSeek/GTRN/SignRep rediscovery not new evidence. No new assets, training or
benchmark TEST data access/evaluation. No Q38/Proposal8/GO or global barrier.
Goal active, no worker active. Next work needs new admissible mechanism evidence,
not repeated availability checks or another broad search counted as experiment.

### Completed SEDS temporal input contract

Read evidence/autonomous_search/SEDS_temporal_contract_result.md. Enabled clip-index
loss assumes pose/RGB windows correspond. PH TRAIN path traced factory→loader→
collator→training call→get_sign_output→loss. RGB consumer uses stored feature order
without timestamps; pose windows reconstructed from filtered frame list. Mask-count
assertion verifies slot validity, not source-frame equality. Paper§3.1 explicitly
describes common clip preparation, so actual misalignment is NOT established.

AST shows RGB loading and pose clipping routines identical across3TRAIN loaders.
No compatible offline producer identified by targeted pinned-code search; no
global resource absence claim. SEDS asset prohibition binding, no producer/feature
download, model/data/annotation/checkpoint/GPU/training/DEV/TEST access or upstream
edits. No artificial corruption fixture: it could not establish real provenance.

No repair experiment or novel candidate follows. After two SEDS source turns,
move to a different mechanism with admissible empirical evidence; do not repeat
optional-KL/temporal-provenance gates or repackage closed alignment/fusion methods.
No Q38/Proposal8/GO or global barrier. Goalactive, no worker active.

### Completed SEDS loss activation gate

Read evidence/autonomous_search/SEDS_loss_activation_result.md. All3 released
training scripts omit rgb_pose_kl (store_true/defaultFalse); top3/top5KL is optional,
not the configured objective. All3 request rgb_pose_match coefficient.4. Paper§3.3
supports broad intended clip-index correspondence mechanism; exact equation/code
parity NOT certified. No optional-KL repair/gradient experiment justified.

AST source check also finds missing freeze_exfusion constructor argument in parsed
namespace and full recipe parser failure on fusion_type under currentPython3.13.5.
OfficialCPython3.10.14 uses different string-choice membership and README recommends
3.10: do NOT attribute this parsing failure to original environment. No full SEDS
startup or model executed. Initial audit assumptions failed twice and are disclosed;
final certificate captures failures plus isolated flag parsing, exit0.

No SEDS assets/annotation/checkpoint/data/GPU/training/DEV/TEST evaluation or upstream
edits. No Q38/Proposal8/GO, no global barrier. Generic fusion/distillation/support
closures remain binding. Further source analysis must target enabled mechanism,
not resurrect optional KL or invent a replacement training pipeline. Goalactive.

### Completed SAN TRAIN exposure check

Read evidence/autonomous_search/SAN_train_collision_result.md. Actual pinned SAN
German TRAIN labels7096rows inspected, not CiCo English C16 substituted. Zero
distinct-caption single-transposition collisions; independent enumeration769199
swaps confirms zero. Max identical raw-caption multiplicity63<localbatch64 makes
empty fallback pool impossible under supplied four-rank/no-replacement/drop_last
configuration. Default n1 semantics verified in current pinned textaugment source,
but SAN omits dependency pin and local package absent: author runtime not verified.

Arbitrary permutation superset114rows/5552orderedpairs/4groups; conditional uniform
B64 augmented-cooccurrence bound.278666%, NOT measured collision/retrieval gain.
Both census executions exit0;340toy reachability checks. No model/tokenizer/EDA
execution, training/GPU/checkpoint or DEV/TEST content/evaluation access.

Both public SAN branches checked untruncated: main82aba9c16files;
master2572f6210files, launcher local negative-table path only. Releases/tags empty,
no pagination. No actual table resource verified in these locations. Broader
private/unlinked asset absence not claimed. Do not fabricate a substitute table.

Decision: two synthetic failure cases do not justify a repair campaign under
checked conditions. Tokenizer/table-generated-negative exposure remains unknown;
generic mining/equivalence closures remain binding. After these two SAN turns,
move to a different mechanism, not repeated census/lookup or augmentation rescue.
No Q38/Proposal8/GO, no global barrier established. Goalactive, no worker active.

### Completed SAN negative source-contract check

Read evidence/autonomous_search/SAN_negative_contract_result.md. Eight fixed
synthetic fixtures execute source AST only: duplicate negatives, unguarded
self-table entries, empty fallback pool and a post-generation mocked-swap
collision verified. Negative-owner indexing passes a two-rank algebraic check.
Actual hard CrossEn confirms multiplicity weighting and tied shared-score
cancellation. Both executions exit0, approximately .843/.851s; second JSON retained.
Real EDA/tokenizer/distributed runtime NOT executed. Actual table exposure,
parameter-gradient harm and retrieval effects unknown. Official ACL2026 SAN
metadata/abstract checked, not full paper or performance reproduction.

No data/checkpoint/GPU/training/DEV/TEST access or upstream modification. No
new Q38/Proposal8/GO. Generic validation is a reproduction control, not novelty;
lexical/hard-negative/equivalence closures remain binding. A real exposure claim
needs actual table provenance and permitted TRAIN preprocessing/sampling, not an
invented substitute. Do not repeat fixtures without materially new evidence.
Goal remains active; no global scientific barrier established and no worker active.

### Completed CiCo channel-loss identity check

Read evidence/autonomous_search/CICO_channel_objective_result.md. Actual CrossEn,
balanceddual_mix.5 config and mixed-score bridge inspected. Four fixed FP64logit
fixtures verify separate four-CE loss = mixed bidirectional CE + mean row/column
negative-log Bhattacharyya agreement term. Maxgradientidentityerror2.78e−17;
independent logsumexp loss error1.49e−14. Both source-check executionsexit0;
secondJSONretained. Existing probe suite90passed1.96s. No model/data/checkpoint/
DEV/TEST access, GPU or training; source extraction only.

Important boundaries: Aclean/Baug training means algebraic mean is not clean
deployment score. Arbitrary opposite channel perturbations need not be realizable
by shared CiCo encoders. C12/C18 retrieval-direction gradients do not test this
channel axis. No harmful-gradient or generalization claim follows from identity.
Primary JMLR2023 ensemble decomposition and NeurIPS2023 learner-collusion sources
checked at explicit sections; generic joint/mixed-loss interpolation is known,
not novelty or automatic improvement. Gradient protection collides with closures.
Do not launch joint-loss/coefficient/gradient-surgery or gradient-census campaign
from this finding alone. No Q38/Proposal8/GO; Q14/Q16 broader questions unresolved.

Previous CMCM turn was progress through a new alternate-branch/release check.
Current work adds an actual source-loss interpretation and algebra/prior boundary,
not a rerun of C12/C18. Next work must add a different admissible mechanism or
new resource evidence; do not repeat this identity, completed audits, or lookup.
Optional closure clarification still unanswered, all closures remain binding.
Goalactive; no global scientific barrier established and no worker active.

### Completed CMCM trained-resource lookup

Read evidence/autonomous_search/CMCM_resource_gate.md. Live API verifies two
branches: master remains5d458719d1da2f082e188cc44705003d919e7e97 with20blobs;
alternate main92ec8470d5c62c44518f3e9825ba54cc837026fb contains only97-byteREADME.
Both recursive trees untruncated; tags/releases empty with no pagination.
No compatible trained checkpoint/integration resource found. Generic torchvision
pretrained encoder constructor is not trained CMCM; pretrained_i3d_path unused.
TMCP learned pre-covariance transformations/BN state cannot be replaced with
raw CiCo or random-module statistics for exposure attribution.

Q35 now has an explicit verified-public-resource boundary; no replacementpipeline,
generic-weight download, activation experiment or training. Do not repeat this
lookup absent changed release/supplied artifacts. This concrete resource dependency
does not establish a global barrier. Other U questions remain unresolved; optional
closure clarification unanswered, closures binding. No worker launched; goalactive.
Further progress must bring new source evidence or a concrete admissible mechanism,
not repeat the completed meta-audits or generic positive-control detour.

### Completed Q01 control adequacy analysis

Read evidence/autonomous_search/Q01_positive_control_adequacy.md. Current trainer,
C04 support and C05 summary reread; six informative coefficient0 runs reach
78.727487–82.580455% T2V /77.103632–83.407145% V2T fixed-TRAIN-pair accuracy,
all selected DEV gains zero. Learning exists; held residual adequacy does not.
Primary ACL sources on selectivity, MI and MDL checked with explicit read scopes:
they measure different quantities, do not upper-bound information from a failed
probe, and cannot retroactively turn fixed-TRAIN/repeated-DEV logs into online MDL.
No injected-ID, corruption, teacher-copying, random-label or MDL campaign launched.
Q01 stays unresolved; generic positive controls do not repair its missing evidence.

Optional user clarification sent: keep blanket Proposal1–6 closures binding, or
permit one new-evidence-backed reconsideration? No response assumed. All closures
and original GO gates remain binding until explicit user direction changes them.
Do not repeat the all37/Q02/Q22/Q01 meta-audits as another progress cycle. Next work
must add new source/data evidence, resolve a concrete resource dependency or receive
a material scope decision. Missing formulation is not a global scientific barrier.
No new model/data/TEST access, DEV evaluation, training, assets or upstream edits;
no research worker launched. Goalactive, no GO or global blocked status.

### Completed all37 scope audit and Q02/Q22 follow-up

Read evidence/autonomous_search/Q01_Q37_feasibility_audit.md. All37 map entries
retain narrow results, broader unknowns, exact evidence links, prerequisites and
closure boundaries. The companion inventory fingerprints explicitly linked files;
it is not a replay of all experiments or validation of every scientific claim.
No new launch-ready high-information intervention is established, but unformulated
questions are NOT thereby infeasible. Neither GO nor global barrier is supported.

Immediate analytical follow-up completed in
evidence/autonomous_search/Q02_Q22_function_class_boundary.md: one shared scalar
matrix can satisfy both directions' singleton ranking constraints, and rank-two
unit-circle scores can have arbitrarily many strict paired maxima. These are
existence counterexamples, not learned/generalizing solutions to actual duplicate
inputs. Different directional recalls or gallery size>dimension alone do not
justify independent heads, higher rank or cohort inference. CiCo's nonlinear
token scorer must not inherit a pooled-dot-product rank bound. Current source
paths inspected; no new experiment, candidate or Q38.

Historical next target, now completed above: Q01 positive-control adequacy without reopening residual
heads, duration/schedule/optimizer/freeze/seed rescue. State what an admissible
positive control could actually establish, its resource lineage and its open
decision consequence. Do not repeat the just-completed Q02/Q22 argument or equate
missing formulation with information absence. Other U rows remain unresolved.
No new model/data/TEST access, DEV evaluation, training, assets or upstream edits
in this audit. Goal active, no GO/global blocker; no research worker launched.

### Completed SignRep comparator check and audit handoff

Q01 comparator check: SignRep official source pinned06f40b5d287867b24e0dd2dc380b40b3f2ae8ac2;
complete12-file tree and four relevant files inspected, no assets/execution.
Paper §§3–7.2 establishes different Hiera/Kinetics/YT-SL pretraining and dictionary
retrieval task. A feature swap would not isolate our information-loss hypothesis
or establish novelty. features and latent differ by learned LN/linear projection;
wrapper computes auxiliary heads even on default extraction path. No harm claim.
AS-C26/C27 and AS-C42 results reread: equal-model/gallery comparison already done,
extended calibration still fails original adequacy; no rerun or recipe rescue.
Read evidence/autonomous_search/SIGNREP_scope_and_feasibility.md.

The following was the handoff before the completed audit above:
Next mandatory work is a current-artifact feasibility/exhaustion audit of ALL37
existing map entries, distinguishing narrow negative probes from unresolved
broader questions and naming concrete permissible next interventions. Do not
invent Q38 just to count another idea, or infer a global barrier from source
limitations. §§10/43 require explicit exhaustion evidence before a terminal
research-barrier conclusion. That conclusion is NOT yet verified. No GPU worker,
new assets, DEV evaluation, upstream edits or training; goalactive, no GO/blocker.

### Completed subword identity resource/prior-art gate

Previous goal turn was progress (completed CiCo census changed prioritization).
New text/annotation-side question: can existing human labels attribute sentence-
retrieval failures to fingerspelling? How2Sign appendix describes FS markers,
but current official download section has no gloss link; latest maintainer
statement found (2022-10-14) reports suspended collection/ongoing subset cleaning.
Closed2025issue24 is a duplicate pointer, not release evidence. No verified
TRAIN/DEV fingerspelling join. FSS-Net(ACL2022) and Tanzer(NAACL2025) establish
prior art for detection+text matching and character-level sentence translation.
Full CiCo tokenizer source contains inverse byte/symbol maps; subwords alone
are not a spelling-loss certificate. No character/teacher/crop branch launched.
Read evidence/autonomous_search/SUBWORD_resource_and_collision_audit.md.
Public issue comments unexpectedly contained unrelated TEST examples; quarantined,
not used or looked up, not retained in artifacts. No TEST dataset file or model
evaluation; no data/assets acquired, upstream edits, or training. Do not claim
zero incidental TEST-related exposure. No current-project bottleneck measured;
this resource route is not promoted, wider science unresolved. Change question;
goalactive, no GO or global blocker, no workeractive.

### Completed CiCo pseudo-clip exposure screen

TRAIN-derived pseudoindex census completed and exactly replayed except PID/walltime,
exit0, .169077s/.167691s. Existing first-anchor grouping behavior was already
documented in OCEM; new evidence is actual local index exposure. Nonlocal gaps
affect68/9854 adaptation-training segments (0.690075%). Equal-segment probability
of a random crop touching recorded-support gaps is0.615417%, below locked10%
gate; fully uncovered crops0.241376%, uncovered-frame fraction0.413895%.
All10992rows map to officialTRAIN7096; all25183possible random-crop starts checked.
Independent frame/interval counts and actual sampler AST agree exactly. No raw
videos/features/checkpoints/DEV/TEST/SEDS/training/upstream edits. Probe-directory
suite90passed1.76s (different scope from earlier combined92-test suite).
Read evidence/autonomous_search/CICO_pseudoclip_result.md. This local P14T index
is not the strong H2S-transfer-aware resource; gap support is not semantic label
correctness. Exposureleadfalse, no repair/teacher/NMS/crop-length/fusion rescue.
Next: change research question, preserve OCEM closure, require an internally open
decision consequence before a new efficacy probe. No workeractive; goalactive,
no GO or global blocker.

### Completed CMCM covariance source audit

New CMCM source audit: learned CSA correction does not itself guarantee invariant
features; publisherpreview not fullidentification theory. No causal-gatingcandidate
revival. Exactcommitted covariance AST on valid PSDfeatureinputs confirms forward
trace+epsilon / backwardtrace mismatch. Sixfixedtraces1e−7..1; source/reference
gradientrelativeerrors99.989,9.999,.99984,.100001,.00100005,.0000100005.
Reference/fullforwardFD agrees<=3.31e−11relative. Isolatedposthoc two-denominator
consistencycontrol restoresgradient<=1.12e−16absolute. Originalexit0,.123270s;
validationexit0,.070721s. Zero covariance forwardfinite/backwardnonfinite.
No upstreamedit/assets/datasets/TEST/training.92tests passed1.79s. Read
evidence/autonomous_search/CMCM_source_result.md. Missing trainedactivation
exposure and runnablematchedpipeline mean no retrievalharm/SOTA/novelmethodclaim.
Next: other internallyopen source-led mechanism; do not build a replacementCMCM
or promote generic numerical repair. No workeractive. Goalactive, no GO.

### Completed UPRet reduction sensitivity screen

Latest controlled TRAIN probe completed and exactly replayed (excluding walltime),
exit0,9.978587s/9.937662s.128 TRAINexamples,32calibration+3×32measurement,
CLIPinit and exactpartialstep767. Fullsum changes medianvisual/textgradients
2.883/3.055% atpartialstate; frozenamplitude sum only.174891/.220684%, both
belowlocked1%leadthreshold. ScalarB0parityexact; usedparametergradientparity
within4.77e−7. No updates/DEV evaluation/TEST; before/afterstatehashesexact.
Read UPRET_real_gradient_result.md for B32/partialcheckpoint/saturationlimits.
One failed engineeringattempt (unusedparityparameter) preserved, repaired before
successfulrun; thresholds/batchesunchanged.90tests passed1.89s. No workeractive.
Next: different repo-led causal layer with an explicit internallyopen decision
consequence; no long reduction campaign, amplitude/optimizer/temperature rescue.
Goalactive, no GO. Historical audit sequence follows.

User explicitly prioritizes CMCM/SAN/SEDS/SLRT/UPRet/CiCo source weaknesses and
controlled improvements toward SOTA. Existing closures/TEST/SEDS-asset restrictions
remain. No claim that source bugs alone establish a novel method or SOTA result.
See evidence/autonomous_search/REPO_LED_WEAKNESS_RESEARCH.md.

New committed-UPRet synthetic audit completedexit0,.737017s:32 realizable unit
embedding matrices, source-AST Sinkhorn/transport block. All32 backward gradients
differ from full-forward finite differences; max.184077, fixed-plan control
error2.22e−11.31/32 finite-difference step checks stable<1e−7; one1.393e−4.
Exact2×2 plan confirms surrogate discrepancy, not solely iterative tolerance.
Paper explicitly fixes the plan and drops the term at inference: neither is an
accidental omission. No harmful-training or deployment-gain claim.

Distinct source/paper reduction discrepancy: committed source uses maxima of
transport-weighted sample similarities, paperEq21 a weighted sum. Posthoc same
synthetic bank:59/496 strict pair-order reversals for max versus sum, also59
using exact plans. Source values independently reconstructedwithin1e−14. No
semantic labels, assets, TEST, fitting, upstream edits or benchmark runs.
Final ECCV paper verified: Eq18 retains weighted sum; sample pooling and K-sample
support are intentional, not a discrepancy. Inference omission is confirmed by
final §4.3; explicit fixed-plan wording above is from arXivv1, not final methodology.
Existing Method1 integration retains max reduction. Read-only readiness census
exit0,.177814s:49 recorded source files match; separate full-tree hash also matches.
82 historical DEV records through1066/2600; best savedmean19.845857, last18.497110.
Latest-train JSON1060, no complete marker; no checkpoint loaded or resumed.
Its feature regime differs from strong CiCo control; no competitive matched UPRet
baseline established. Next: register real-TRAIN-batch combined-loss/encoder-gradient
sensitivity with shared encodings/noise, max/sum/frozen amplitude control before
expensive benchmark arms. No source edits to Method1/UPRet, training or TEST.
See UPRET-READINESS_run.json and REPO_LED_WEAKNESS_RESEARCH.md. No generic OT,
uncertainty, gradient-fix or distillation method promoted. Latest combined baseline
fixtures/probe suite88passed1.92s, exit0. No research worker active; unrelated GPU
process observed and untouched. Goalactive, no GO.

### Completed AS-C45 exact-gloss screen

First runexit0,1.281996s. TRAIN6900 unique gloss sequences,70repeatedgroups266rows,
843pairs,345different-native-and-model-text pairs. DEV512unique,5groups12rows,
9pairs,4different-both pairs. Persistent strict same-gloss/different-text confusers
acrossall3seeds:1/92T2V,2/87V2T; bothbelowregistered10%threshold. Leadfalse.
Independent all-pairs census exact; separate scalar score validator exactexit0,
.196058s;81tests passed3.14s before new UPRet fixture tests. No fuzzy-gloss/miner/
equivalence rescue. RunSHA6e3c912ef8a11eb14201579810cb0fa79cffddaf48f1639842174d9300176163.
See evidence/autonomous_search/AS-C45_result.md. Worker376407terminal.

### Completed post-C44 direct annotation-join audit

Newly read local PH2014T release README explicitly states that the ECCV2014
manually annotated mouthing sequences are excluded from ALL sets. It also
documents cross-release split separation and changed sentence boundaries. The
original CVPR2018 dataset paper §4 corroborates boundary changes and separation
within the same author lineage. This is documented provenance, not a recomputed
pixel-level overlap certificate. Do not download those archives for a direct
joint-cue join or infer linkage by removing filename suffixes.

Metadata census completed first execution,exit0,.076262s: seven files hashed
(two READMEs, standard TRAIN/DEV CSVs, complex TRAIN CSV, two manifests).
Standard7096/519 IDs and native captions match manifests exactly; complex7096 IDs
match TRAIN, byte-exact captions0 (not semantic disagreement). All14711 CSV rows
have start/end−1/−1 and sentence-local frame globs. Independent CSV/field parsers
agree on every field. No TEST file, annotation pickle/gzip, image/video/feature,
checkpoint or score opened; no archive acquired, no training or new annotations.
Result: evidence/autonomous_search/POST_C44_annotation_result.md.
RecordSHA7abb19d8626177460f78ff6af441b97862c342210f1a5ec20703217101c7abae.

Decision: close this direct published-mouthing-label join, not all simultaneity
science or other potential resources. No pseudo-label/filename/new-benchmark
rescue. Move to a different testable open causal mechanism; no further archive
join work pending. Optional expert question remains unanswered but does not block
the global search. No AS-C45 registered or worker active. Previous and current
goal turns made concrete progress; goalactive, no method GO or global exhaustion.

### Completed AS-C44 — retained structural scope correction

Switched from gallery filtering to Q17 relation/intervention validity. Registered
CPU architecture/synthetic certificate completed first run, exit0, .098776s;
no dataset/features/checkpoints/videos loaded, zero training updates.
R2/R3 raw readouts are permutation-invariant after I3D: three permutation types,
max FP64 discrepancy5.55e−17, nonzero/content-sensitive controls pass. Their
negative AS-C02 results do not directly close explicit spatial/window arrangement.
The cached four cells are not isolated articulators: adaptive7→2 bins overlap and
each has full224-pixel axial structural support; temporal support spans16frames.
Nominal max RF99time/379spatial, not effective gradients or anatomical labels.
Set/Boolean propagation and torch adaptive basis checks agree. No joint3D voxel
enumeration or real linguistic information-loss claim. 79pytest tests pass1.83s;
initial wrong unittest discovery had30 import errors, disclosed in result.

Primary source check finds synchronization/multi-cue prior art and manual handshape
and mouthing resources, but no verified local joint-contrast mapping. Handshape
page3359 vs paper3361 count discrepancy unresolved. Model-generated alignments
are not expert labels. No resource archive acquired; PH2014T README link404.
No generic positional/stream/synchronization/RCLI or RF-weighting rescue.
Results: evidence/autonomous_search/AS-C44_result.md and AS-C44_sources.md.
RunSHA7a184b678aadb75008e78870e6f6324a63bbef48e3e0ccf2c7c4a2b35473dcb1.

Next: safely establish manual-resource TRAIN/DEV provenance and joint-contrast
validity if it can support a distinct open mechanism; otherwise switch question.
Optional expert-annotation question unanswered, not a global blocker. No active
worker or AS-C45 registration. Goalactive; certificate is not a GO candidate.

### Completed AS-C43 — retained gallery-filter result

All16registered condition pairs completedexit0,15.267749s,peak1.590GBGPU.
Fixedgallery smoothing:0/3seed gates; sharpening:0/3. Uniform3ensemble smoothing
70.712909 vs77.263969 (−6.551060pp,CI[−8.728797,−4.536675]); sharpening76.685934
(−.578035pp,CI[−1.679104,.496056]). Sharpensemble T2V−1.541426/V2T+.385356;
persistent ranks worsenboth. Bothdiagnostic_leads=false,method_go=false.

Independent validator completedexit0,4.814617s: all32matrices reconstructedEXACT
by candidate/neighbor loops or ensemble sums;96officialrecalls andsavedranks exact.
All3fresh baseline encodings/scores exactly replayed before interventions; every
single-query vsbatched operator exact. Sixnearest-neighbor boundarycertificates
within1e−12cosine, graphrelabelingexact. No independent fullencoder rerun claim.
77tests passed3.71s. No activeworker; worker330237terminal. Results:
evidence/autonomous_search/AS-C43_result.md. RunSHA
6926ee09ae74edc6509f38dbf5377d407da76db4350522fa3cbfe5c9e518bfed;
validationSHAe5c6b2155d0f91d50dd610e59948c651fb0d416fa7ed3b7e8f199d5a131714cf.

No graphsize/strength/layer/temperature/learnedgraph rescue. Known graphfiltering
is not novel; lessdamage than shuffle is not a gain overbaseline. No otherqueries,
newlabels, testaccess, fitting or newresources. Move to a differentcausallayer with
an explicit open decisionconsequence. Currentgoalturn made concreteexperimental
andvalidation progress. Goalactive; no GO, globalexhaustion or blocker.

### Historical AS-C43 registration — experiment now terminal

AS-C43 tests fixed five-neighbor gallery structure with smoothing/sharpening and
node-shuffled controls, all3historical checkpoints plus uniform3ensemble. No
other queries, fitting, test data or changed positives. Known graph filtering is
NOT a new method; this is a §§13–15 competitor-feature diagnostic. Protocol and
implementation saved; 3fixture tests pass. No outcome yet. Check actual worker
and AS-C43-GALLERY_run.json for liveness/status, not this registration paragraph.
Previous goal turn made progress by completing the discourse boundary audit.
Goal remains active; no method GO. See AS-C43_protocol.md for fixed decision gates.

### Completed post-C42 discourse boundary audit

Switched away from calibration to external discourse/context availability.
Pre-candidate audit (NOT AS-C43) read/hashed 7619 PH TRAIN/DEV metadata files.
All7615 temporal records describe sentence-local input frames, not verified
recording offsets. Numeric predecessor counts: TRAIN5617same/312other/1167absent;
DEV117same/292TRAIN/110absent from inspected TRAIN/DEV. Independent string
enumeration agrees exactly; 3tests pass. Inferred adjacency is NOT verified
discourse continuity. No TEST, video/features, checkpoint, scores or model run.
Full probe regression suite:74passed1.47s,exit0.

Five-primary-paper scan plus official PH documentation: preceding-context
translation and coreference mechanisms already exist; How2Sign human-context
findings do not establish PH retrieval irreducibility. Generic concatenation lacks
verified task-compatible linkage; context-decoder/protection collides with RPCA;
source-conditioned calibration is not a new mechanism. No candidate promoted.
See evidence/autonomous_search/POST_C42_discourse_boundary_audit.md and sources.
Metadata result SHA3834eb7c776279176564d5bf54aa11f6ab3c27c725b552e2d18cd4d238864c2b.

Next: select a different open mechanism with internal-closure and existing-input
testability checks before any new diagnostic/pilot. No neighbor-caption or
linkage-rescue sweep. No AS-C43 registered/launched. Goal active; no GO, global
exhaustion or blocker. AS-C42 remains completed/inadequate as summarized below.

### Completed AS-C42 endpoint (retained authoritative experiment result)

Attempt2 completed8800updates/200passes,exit0,3816.357260s,peak14.129GBGPU.
FinalheldR1T22.545455/V22.400000,mean22.472727,versusshort17.490909(+4.981818pp).
FinalfitT96.145455/V96.363636;heldstrictdifferent-text errors1042/1037. Bothheld
directions remain belowORIGINAL50%gate: diagnostic_adequacy=false. Relativegain
and manyerrors do not authorize calibrated residual supervision. Jointbudget/
stretchedschedulecontrast, not pureexposure or upstreamPHrecipereproduction.

Finalvalidator --attempt2 completedexit0,10.775077s: all8matrices officialrecalls,
perqueryranks and strictresidualindexesexact; independentadequacyagrees; final
held1375x1375scores replayEXACTfromcheckpoint (maxerror0). All264originaltraining
summaries/3evaluationrecords match recoveredprefix; no unsavedparameter/optimizer/
RNGstate guarantee. Originalinterruptedattempt preserved, causeunknown. No active
training/validator process remains; all liveness snapshots below are HISTORICAL.
71tests pass1.78s. See AS-C42_result.md. SuccessfulrunSHA256
08dc062979c5235b78b76219062f5f902de842c2a1a1ab6df0ec8404114ad191;
validationSHA25694f5d9127d8a1a57ce94174ce04d679cb211d10b515f1b7409753d57ecd4acbe.

No calibration GO, methodcandidate, informationceiling or globalexhaustion.
No duration/schedule/optimizer/freeze/seed rescue sweep; Q01remains unresolved.
Next: materiallydifferent mechanism with explicit open decisionconsequence,
not mining this inadequate model's errors or simply changing initialization.
H2Sbyteprovenanceverified buttraininglineageunverified. NoAS-C43registered/launched.
Previousgoalturn: replaymilestoneprogress; currentturn: completedtraining and
independentendpointvalidation, authoritative mapupdated. Goal remainsactive.

### Historical AS-C42 interruption/recovery snapshots — not live instructions

Latest recovery milestone supersedes older snapshots: attempt2worker195622 alive
at16m04s, reached2200updates andfirstevaluation. All4arrays/filehashes at0/2200,
first88traininglogs and2completeevaluationrecords independently compared EXACT
to interruptedattempt,exit0. Fitmean94.509091/heldmean14.0 reproduced. No new
efficacy estimate, checkpoint selection or unobserved state-tensor parity claim.
Next evaluation4400, final8800. Full264-record interruptedprefix and finalgate
remain pending; no restart or changed protocol. Currentturn: verifiedmonitoring
plus a completed numerical replay milestone, no GOorblocker.

Current continuation: actual attempt2worker195622/timeout195592 alive at3m12s,
425updates;17trainingrecords andinitialevaluationexact. Matching scope is logged
numerical summaries and evaluationrecords/fullscorehashes, NOT unsaved every-step
parameter/optimizer/RNG states. No new endpoint or changed protocol. This turn
is verified monitoring plus independent original4400metric validation: all12
R1/R5/R10valuesexact to production evaluator,exit0. HeldR1T18.763636/V16.8;
fitT95.781818/V96.0. Bothheldbelow50% but final8800gate notyet evaluated.
See AS-C42_progress.md for original4400scorehashes. Continue sameworker; do not
start another attempt or finalvalidator while training remains live.

Authoritative recheck supersedes ALL older liveness snapshots: original140715
and controller absent, GPU empty, exec15558unknown; originalJSON stale-running
at6600updates/2855.58seconds, no traceback or finalcheckpoint. Cause/exitcode
unknown; accessible kerneljournal empty is inconclusive. Preserve originalJSON,
console and6matrices unchanged. NOT completed/failed scientific condition.
Step4400 fitmean95.890909/held17.781818 is intermediate, not a selected endpoint.

AS-C42_interruption_recovery.md discloses attempt2 from identical initialization
because no resumable state exists. Originalprotocol/harness unchanged; separate
wrapper changes ONLY3output names across6ASTconstants, inverseASTexact. Every
available training/evaluation prefix record must match interruptedattempt;
all264training/3evaluationrecords required for finalvalidation. No seed/budget/
schedule/loss/precision change, no performance-selected retry. Durable detached
process session reduces observation dependence; unknownterminationcause remains.

Recovery launcherexit0, timeoutcontroller195592. Use explicitattempt2launch/
worker/exitJSONs and actualprocesses; original15558is not a livehandle. Final
validator prepared for --attempt 2, must await completed8800endpoint. No GO.

Public CiCo archive byte audit completed exit0,60.894018s: archiveSHAexactold
provenance; H2S andPHmembers exactlymatchlocalfiles. RecordH2S-RELEASE-MEMBER-
VERIFY_run.json; retained971MB archive underphase2/H2S-release-audit. H2S training
lineage/PH-unfittedness remainsunverified; no resource substitution or inference.
Current turn progress: interruption established, disclosed recovery launched,
public-release byte integrity verified. Notblocked, notgloballyexhausted.
Latest actual attempt2 check: controller195592/supervisor195593/worker195622
alive at53seconds;100updates, initialscorearrays/hashes/evaluationexact, first4
trainingrecords and1evaluationrecordexact. Full6600prefixnotyetchecked.71probe
tests passed2.91s. RecoverywrapperSHA256
7c32addfc7fc8b0e41e144a2782b551b4888ec35d3ca6aa7cd6d4890c0763b7a;
interruptedrunSHA256eb0c7157bb5d5db655322db258ae6c93b34f41a5f2a1d5d4e88202f3c639ca12.

### Historical pre-interruption monitoring snapshots (NOT current liveness)

Latest actual liveness check: same workerPID140715, execsession15558, elapsed
17m11s and2350/8800updates. First fixed evaluation2200: fit T2V/V2T R1
94.036364/94.981818; held14.545455/13.454545, mean14.000000. Held remains below
the original50%-each gate at this intermediate step. No final adequacy decision,
early stop or protocol change. Both saved matrices independently checked via
production evaluator: all12 R1/R5/R10 cells exact, exit0. Harness and five helper
hashes unchanged. See AS-C42_progress.md for hashes and11/11interpretation scan.
Next fixed evaluation4400, final8800. Original training still live; do not
start the final validator or restart a completed/unknown observation handle.

Auxiliary read-only resource audit: artifacts/pretrained/H2S_sota.pth exists,
SHA25667f557976d523dc3cd63cae73c29c88145e409b77221c90260ae51b9d0c95b35;
305tensor layout matches PH,299tensors differ, no training metadata. Local
provenance binds PH member only, so H2S training lineage/PH-unfittedness remains
unverified. Initial false absence from a bad filename filter explicitly corrected.
No resource substitution/download or new training run. See auxiliary checkpoint
audit; it is not a clean teacher, candidate or AS-C43 experiment.

Current goal turn: verified monitoring plus new intermediate retrieval evidence
and independent metric checks. No GO, global exhaustion or blocker.
End-of-turn live check: samePID140715 at18m19s,2500updates,epoch56; finite loss,
RSS4.81GB. No worker restart; next checkpoint remains4400.

AS-C42 tests whether a larger fixed learning budget supplies an adequate
TRAIN-only residual probe for Q01, not a method or upstream reproduction. Same
immutable AS-C20 program, all trainable parameters/optimizer/peak learning rates/
batch128/seed42/data/augmentation/FP32 unchanged; total8800 updates (200 passes),
warmup880 and proportional cosine schedule, fixed evaluations0/2200/4400/8800.
This is budget PLUS stretched schedule, not identical-prefix continuation or
pure exposure attribution. No table-freeze/optimizer/seed sweep or lowered gate.
Both held R1 must still reach50%, both fit80%, mean gain5pp and50 strict
different-text held errors each. A pass only permits further residual diagnosis;
no closed teacher/correction family, method GO or R0-equivalence claim follows.

Protocol AS-C42_protocol.md; harness extended_budget_calibration.py reversibly
changes16 registered AST literals. Metrics/adequacy/evaluation functions remain
AST-identical (the metric percentage multiplier100 is not warmup100). The short
control already has exact full AS-C30 replay; no redundant control added.

Launched command: PYTHONPATH=shared:. timeout7300 python -m
methods.information_probe.extended_budget_calibration (environment Python is
/home/haipd/miniconda3/bin/python; actual shell command has `timeout 7300`).
Exec session15558, timeout controllerPID140714, workerPID140715. Latest actual
process check alive at1m52s; JSON225/8800 updates at102.54s, finite logged loss.
Before any update, both1375x1375 initialization score arrays AND file hashes,
and full evaluation records match AS-C20 EXACTLY. No endpoint outcome yet.
Durable console artifacts/proposal7/phase2/AS-C42-BUDGET-console.log; progress
AS-C42-BUDGET_run.json. Internal7200s/external7300s hard limits; monitor30–60s.
Do not restart from an expired observation handle; inspect this worker and JSON.
Later live check: same workerPID140715 alive at2m48s; console375updates,
epoch8, finite loss. Full probe suite70passed in3.10s; compileall/diffcheck pass.

Final validator prepared, NOT executed: validate_extended_budget.py refuses
incomplete training, verifies all8 matrices against official metrics/ranks and
strict residual indexes, independently recomputes adequacy, checks parent/helper
hashes and replays final held scores from saved checkpoint. It does not rerun
optimization. No DEV/test data or PH-fitted retrieval checkpoint in this run.

Previous goal turn: progress (AS-C41 finalization and closure correction).
Current goal turn: progress plus verified live training, not a completed result.
No method GO, Proposal8, global exhaustion or blocker claim.

### Historical AS-C41 completion and boundary correction

AS-C41 completed exit 0 in 2.457207 seconds. Its preregistered one-word edit
intersection has 396 TRAIN pairs (386 rows, 270 joint templates), all with four
strictly positive baseline margins; minimum 1.268291. DEV has 0 English and
9 German one-edit pairs, hence 0 bilingual pairs. This is missing DEV support,
not successful generalization or a numerical paired-kernel validation. Native
wording corroboration is not expert-validated signed-language minimal contrast;
some pairs are near-synonyms. No edit-distance/native-only rescue or method GO.

AS-C41-INVENTORY-VALIDATION completed in 0.598492 seconds: independent brute
force enumeration agrees exactly on pair counts, IDs and differing positions.
Scores and encoders were not rerun by that validator. See AS-C41_result.md and
AS-C41-SINGLE-WORD_run.json, SHA256
608c17b0ddc1cc7ed1f20c55638e48756f3f96614c9475f47685e92a84e5639b.
No active experiment or validator; do not restart completed work.
Finalization check: 67 probe tests passed in 1.76 seconds; parent-run hash
matches the independent inventory validator. No new score or encoder replay.

Post-C41 boundary audit corrected Q13: signer/nuisance invariance, including
selective removal, is already CLOSED in the proposal/negative registry. AS-C10
also already performed signer-stratified diagnostics. A generic signer probe
must not be treated as an open route to a renamed invariance method. See
POST_C41_boundary_audit.md. Next selection must state which materially open
method decision a measurement would change, before another experiment. No
AS-C42 registered/launched, no Proposal 8, and no global exhaustion claim.
Previous goal turn: progress (new census and validation); current turn: progress
(terminal evidence checked, result recorded, misleading search-map entry fixed).

### Historical AS-C40 completion and pre-C41 selection snapshot

Current turn progress: allsix matched ON/OFFruns completed260updates each.
AS-C40-VALIDATION completed exit0,2.897022s; all18saved matrices official-checked;
initialstate,config,batchorder,clean/visualinputs,scheduler andinitializationscores
match within each seed. OFF−ONmeanR1:42−.385356CI[−3.254438,2.446235],
1337+3.660886CI[.297604,6.866538],2026−1.541426CI[−4.696146,1.530612].
0/3passallregisteredgates; favorable1337failsT2VR5retention(−.578035pp).
EveryOFFendpoint below commoninitialization74.759152,by7.707129/5.394990/
8.092486pp respectively,also belowstrong42andAS-C32ensemble. NoGOoraugmentation
removalmethod. Do not sweep strength/schedule/precision/seeds to rescue it.
Sixsuccessfulruns1577.354959s,max43.260GBGPU. Failed2026ONBrokenPipe attempt
preserved(172.416172s); durableattempt2exactlyreplaysfailedprefix thencompletes;
2026OFFcomplete. Finalcontrollerexit0, noactiveexperiment.65tests passafterrepair.
See AS-C40_result.md and AS-C40-VALIDATION_run.json (SHA256f43eb75ffd00cb600fc1cb0013e387fcb6beebce9be2240beda8681cdb028373).
Next: re-rank remaining open questions using this learning-intervention result
and move to a materially different layer; no AS-C41created/launched. No global
exhaustion orProposal8. All queue/PID statements below are HISTORICAL snapshots,
not current liveness instructions; do not restart any completedcondition.

Next feasible lead for reprioritization:Q09/D (natural lexical contrast support),
distinct from failed optimization intervention. AS-C16tested SAMEwordmultiset
order differences, not naturally occurring one-word substitutions corroborated
by native captions. Inspect existing TRAIN/DEVannotation/manifest fields first,
then preregister any inventory before measuring outcomes. Distinguish native/
translated wording evidence from expert-validated signed-language minimal pairs;
no new annotations, negatives, benchmark, labels, ELSClexicalmethod or translation
preservation revival. Do not infer a bottleneck from mere pair existence. This
is an untested question, not a candidate or currentAS-C41execution.

### Historical AS-C40 monitoring and recovery record

Live-state recheck supersedes older queue snapshot below:42ON/OFF and1337ON/OFF
allcompleted. Original2026ONfailed at169updates/epoch12progress print with
BrokenPipeError; no training/numerical failure, processqueue ended,2026OFFnotrun.
Failed2026ONrecord/folder retained. AS-C40_infrastructure_repair.md registers
2026ON-attempt2fromsameinitialization and then2026OFF, durable per-run console
logging. No training code/seed/horizon change; optional outputattemptsuffix only.
New controller launched after2tests/compileall/diffcheck; inspect actualprocess
and JSON before resuming. Do not restart the original completedfour conditions.
Validator now uses explicit2026ON-attempt2name and verifies18matrices, configs
differonlyaugmentation, all260batch/forward/negativepoolcounts. No gate evaluated
until allsixcomplete. No GO. Previous turn classified progress+verifiedwait.
Latest verifiedlive controller:execsession21612,PID78021;2026ONattempt2worker
PID78023, eightepochs completed at103.29s snapshot, console grows normally.
Paired integrity checks for42and1337 pass(initialstate,batchorder,clean/visual
inputs,scheduler,initializationscorehashes); partialOFFminusONmeans−.385356pp
and+3.660886pp respectively, bothOFFbelowinit. No cross-seed gate calculated.
Removing three output-only code additions exactly reconstructs original worker
SHA256c6746557d34bed057db8faa6af81ff3b5f46d826021a411f407dc547b92a1201.
Later livecheck2026ONattempt2reached13epochs/169updates at167.07s; EXACTLY
matches failedprefixinitialstate,169batchtraces,13evalrecords and181logs
excludingtime/memory.65tests pass3.04s. No worker restarted again;2026OFFqueued.

Current turn progress: AS-C39 original seed42epoch0/13updates replay passes
EXACTLY:312model tensors,302optimizer parameter states,scheduler,AMPscaler,
allRNG/sampler state,trainable names,13traininglogs+evaluationlog andfullscores.
Exit0,16.666957s,43.260GBGPU. Archivedtrain function from historicalelsc/train.py
compiled unchanged; active dependencyASTs verified. No initialevaluation added,
original200epochschedule preserved. No checkpointwrite. See AS-C39_result.md.

AS-C40 preregistered six matched runs:3seeds×augmentationON/OFF, fixed260updates
(20epochs,endoforiginalwarmup) with original200epoch schedule. Baseline-only,
same2textforwards/objective/compute, OFFchangesonlycollatoraugmentationconfig.
Trace initialstate,batchIDs,clean/visualinputs,auginputhashes andcounts without
RNGdraws. No epoch/seed/policy selection, no genericaugmentation-removalGO.
First run AS-C40-s42-on COMPLETED exit0,256.934556s,43.260GBGPU:
endpointT67.630058/V67.244701,mean67.437380; initializationT74.181118/V75.337187.
All280logs (260training+20DEV) match historical42first20epochs exactly excluding
time/memory.133120trainingrow occurrences,65224augmented. Firstepochfullstate
andscorechecks exact. No augmentation effect inferred without matchedOFF.
Remainingfive queued sequentially in execsession9831:42OFF,1337ON,1337OFF,
2026ON,2026OFF. This controller stops on any worker failure. At launch42OFF
is active, laterfour are queued not yet started. Inspect actualrunJSONs/process
before resuming; never restart on an expired observation handle alone.
Protocol AS-C40_protocol.md. Unit tests3passed prelaunch; fullsuite64passed
before additional validator unit test (2validator/trace tests also pass).
After allsixterminalcompleted, execute validate_augmentation_training.py;
it refuses incomplete inputs and checks matchedbatch/initialstate/inputs/
schedule,unchangedOFFtexts,officialmetrics and allregisteredperseedgates.
Latest live check:controllerPID27171,42OFFworkerPID27173running; two epochs
completed in32.36s, no failure. Subsequent runs pending behind that worker.
Do not mistake this active experiment for a completed200epochtraining campaign.
No GO; actual augmentation training effect still unmeasured pending sixruns.

Previous goal turn: progress (AS-C37 complete inventory). Current: progress,
AS-C38 fixed nativeFP16tail-padding control complete exit0,14.968374s,
1.798GBpeakGPU; all3original score matrices exact, padded encoding/scoring
repeats exact,8matrices official-metric validated.61tests pass1.61s.
Ensemble ALLquery ranks unchanged,77.263969mean,+2.023121vs padded42,
CI[.465506,3.619048],allAS-C32diagnostic gates survive. Single42/1337means
unchanged;2026mean74.855491vs74.951830,oneV2Ttop1loss. All5single-model
rank changes are WITHINunchanged positive ties: positive score,tiedmembers,
strictlyhighercandidate sets unchanged.512x512scores exact; changed tail
columns affect official unstable tie order elsewhere, not strict preference.
Do not call this semantic harm or alter official metrics. See AS-C38_result.md.
No active experiment, no GO, no numerical/precision/evaluator rescue method.

Next selected Q21/layerG action: establish an exact original-recipe first-epoch
training replay before an isolated augmentation-on/off intervention. Existing
AS-C08inference perturbations and AS-C12/C18gradient diagnostics do not measure
actual learning harm. Use baseline-only path in methods/elsc/elsc/train.py,
not any closed ELSCobjective or SEDS/dive trainer. Original42implementation
sidecar points to commit39449e18def6b154944ceeaed39dfd5c570882a3; inspect version
differences and optimizer/scheduler/RNG state before registering a replay.
Preserve original200epoch schedule even if checking first13updates; shortening
schedule would change the intervention. Require baseline epoch0score/state/log
parity where archived outputs permit; no claim of replay from approximateR1.
No generic augmentation removal as novel candidate, no optimizer rescue sweep,
no clean-scratch AS-C20adequacy workaround, no dev-fitted correction. A measured
training effect would still need3distinct derived candidates and novelty review.
AS-C39 now completed with exact replay; AS-C40above is the registered follow-up.

Earlier turn: progress. AS-C37 completed exit0,18.840844sCPU: zero within-clip
valid-slot exact repeats atFP32/nativeFP16 across447139valid slots,7615rows.
TRAIN417323slots/12577248eligible pairs;DEV29816/888354; no repeat-bearing rows.
AllAS-C36manifest/source/input hashes and15230file hashes match. Zero conditional
source-reload/pair-equality checks because no repeats (not a positive replay).
One pre-run synthetic float32-equality fixture failed and was corrected;
3new tests passed before launch. See AS-C37_result.md. No active experiment.
Final full methods/information_probe test collection:59passed in1.59s;
compileall,gitdiffcheck and saved-row/group/count consistency checks pass.
AS-C36/C37 now exhaust these TWO bounded exact-collapse checks, not acquisition
or information sufficiency generally. Switch causal layer; no near-similarity,
sampler or precision rescue sweep. No GO and no global exhaustion.

Next selected layerJ question (Q29): independent-query numerical batching
stability, distinct from input FP16 quantization tuning. AS-C31 observed
same-input text-token changes between full128 and tail batches but did not
test resulting official ranks or AS-C32ensemble gain. Preregister a fixed
batch-shape control (native precision unchanged, pad tail to full batch with
discarded duplicate filler rows, no new gallery candidates) against exact
original128/tail replay, all3models and their fixed mean. Audit full-gallery
ranks/ties, not only tensor deltas; keep official positives and score blocks
fixed. This is computational validity, not a novel batching method or a
precision/threshold sweep. AS-C38 now completed above; this historical plan is
not pending and must not trigger another run.

Previous turn classification: progress (AS-C34 validation and AS-C35 coordinate control).
Earlier turn: progress (AS-C36 all7615TRAIN/DEV input hashes, FP32andnativeFP16,
valid masks included; zero exact collisions in either precision/any split scope).
No repeated aware/agnostic paths, no all-zero valid inputs or nonzero padding.
15230source-file hashes recorded. Exit0,10.36sCPU;62focused tests pass.
No active probe; see AS-C36_result.md. This is NOT a semantic-information ceiling
or proof of disjoint raw footage. No row/label change. Next within-video temporal
slot exact-repetition inventory was subsequently completed as AS-C37 above.
All18AS-C34 score matrices independently metric-checked;
AS-C35 TRAIN-only orthogonal control completed exit0,25.03s,2.433GBGPU.
Aligned mismatched6=76.011561 vs original76.107900,−.096339pp,
CI[−1.183432,.956023],coordinate-explanation lead FALSE. All diagonal ranks
preserved with score errors<=5e-5, identity9cells exact. Anchor agreement improves
TRAIN/dev, but not retrieval; shifted-target mismatch28.227360.60tests pass.
No active probe; do not restart. See AS-C35_result.md. No alignment rescue sweep.
AS-C34 mismatched6 mean76.107900 vs matched77.263969,−1.156069pp,
CI[−2.190476,−.188324], pairing signal TRUE. All3diagonals/repeats exact;
exit0,15.19s,2.116GBGPU.59focused tests pass after safe test rerun (expired
test-session handle; experiment not restarted). See AS-C34_result.md.
AS-C32 uniform score ensemble77.263969 vs best single
75.240848,+2.023121pp,conditional95%CI[.465506,3.619048],all diagnostic gates
pass; AS-C33 uniform weight average76.011561 fails ensemble retention and
regresses V2T vsR0. No GO: ensemble is ordinary3-model control, soup is known
prior-art control, not3independent new-method seeds. Both runs exit0; no live
process.57focused tests pass. AS-C33 inherited bootstrap scope string corrected
in a separate hash-bound annotation; numerical outputs unchanged.
AS-C31 text-prefix inventory/score decomposition completed:
1/92T and0/87V persistent queries meet broad-burden signature, original20%each
threshold fails. Post-hoc batch-shape numeric clarification completed, no retry.
Both AS-C31 runs exit0; no active process.54focused tests pass.
AS-C30 preregistered single-factor freeze comparison:
control replay passed all8 score hashes/arrays,40 training entries,evaluation
records and305 final state tensors exactly; exit0,444.78s. Freeze completed
exit0,452.34s: heldmean18.254545 versus17.490909,+.763636pp, but original
adequacy remains FALSE. Independent16-matrix metric validation and freeze
checkpoint inference replay passed exit0,13.84s. No active AS-C30 process;
do not restart either arm.51focused tests passed. See AS-C30_result.md.
Two harness preflight test defects corrected: wrong OUT import before training;
inverse-label collision in test only. Control inadvertently launched before
checking second test failure; node-wise corrected test passes, harness unchanged
during execution. These are not failed training attempts or hidden retries.
AS-C20 fixed fit/held split, no
dev/test or PH-fitted retrieval checkpoint. AS-C07 candidate B is rejected for the current
specification: OTTER already supplies teacher similarity → joint balanced
matching → contrastive soft targets. Out-of-fold provenance is not itself a new
mechanism. Differences and source-access limits are in `AS-C07_collision_screen.md`.
Clean internal-fold calibration completed, not a method pilot; no GO or global exhaustion claim.

AS-C08 exact clean encoder and dev-score replay passed. Deployed augmentation
changes3461–3502/7096 train and252–259/519 dev text inputs. In training-style
A-clean/B-augmented mixed scores, different-clean-input nonpositive fixed train
margins remain0 in T2V and0–1 in V2T. Dev mean R1 is74.470135/74.951830/75.144509
for augmentation seeds42/1337/2026 versus clean75.240848. Both-channel augmented
stress is74.373796/73.988439/74.951830. These are perturbations of ONE frozen
backbone, not three trained methods or proof removing augmentation helps.

Current best candidate: none. AS-C13's aligned maps explain38.88% (pre→post) and
39.51% (post→pre) of dev normalized variation relative to the train target mean;
train values38.82%/40.08% show this probe transfers. Shuffled maps explain less
than zero. Full-gallery mean R1: identity75.240848, forward53.949904,
shuffled3.853565, mean3.853565, roundtrip65.510597. No substitution improves
persistent ranks. These show transferable linear structure, NOT discarded
linguistic information or an information ceiling. No new distillation candidate.
AS-C14 separates video CLS, video padding and text padding exclusions; all seven
nonidentity combinations lose mean R1. Single-factor losses:.385356pp CLS,
.481696pp video padding,1.445087pp text padding; joint1.541426pp. Thus the earlier
joint loss cannot be attributed to video padding alone. This closes the fixed
inference-mask factor screen, not training-consistent policy or gradient paths.
AS-C15 confirms all assignment edges admit additive NON-STRICT support from
label-free dual potentials, but those potentials require the evaluation cohort
and do not reproduce independent argmax tie-breaking. Tiny-noise cohort gains
average6.929994pp; all60 perturbations beat their baseline by≥5.587669pp.
All712 changed-row occurrences swap identical deployed text inputs (post-hoc
description, not semantic validation). Thus headroom survives but does not prove
a nonadditive independent-query expressivity defect. No three new candidates
are justified from that stronger claim. Do not relaunch fixed-bank corrections,
standalone permutation losses or OTTER-like distillation.
AS-C16 finds only2 natural same-lexicon/different-order train groups,19 rows,
61 pair instances, and no dev group. They move “now”/“usually”, not clearly
bound entities/events; all61 already have four strict paired margins. No valid
relational supervision or new text method follows. AS-C17's three-view mean
loses.481696pp mean R1; effective view diversity is only1 for475/519 videos
(including335 changed videos). The fixed low-diversity sampler fails, not all
temporal acquisition. Persistent ranks improve slightly despite lower R1.
AS-C18 completes the video-parameter check: all6 directional cosines positive
(.116255–.203030); exact8-example direct/chain smoke and encoder token parity.
Duplicate-video path signed projections vary.156002–.817325, CLS/pad paths
.048181–.363366. These are video-coordinate pathways, NOT duplicate-query loss
attribution or measured harm. Complementary vectors are algebraic residuals,
not independently differentiated FP16 closure checks. No gradient-surgery or
population-risk candidate follows; PMGR/RPCA remain closed.
AS-C19 makes a PH-unfitted retrieval initialization executable:300 generic CLIP
tensors copied exactly,3 explicit random tensors and2 scalar-one constants;
seed42 repeat exact and1337 changes random tensors. No PH retrieval checkpoint
loaded. TRAIN-only source hash split5721fit/1375held, source overlap0;72held
rows share exact fit text. Finite32-row float32 forward/backward, no update.
This is not proof of zero generic-pretraining overlap or useful residual signal.
AS-C20 completed bounded TRAIN-only learning/calibration under this fixed
internal partition. Protocol fixes 1,000 updates, batch128, float32 and evaluation
0/250/500/1000. Fit diagnostic uses a preselected1,375-row fit subset to match
held gallery size. Learning gate: both fit directions>=80%, both held>=50%,
held mean gain>=5pp; residual gate>=50 strict different-text-input held errors
per direction. These are practical adequacy thresholds, not equivalence to R0.
Result: final fit95.272727/96.290909 and held17.090909/17.890909 R1(T/V);
held error counts1120/1102 pass count threshold but learning adequacy FAILS.
446.31s,14.13GB allocated; all8 evaluations match official metrics, final held
score replay exact. No training replication; expired session exit not recovered,
terminal completion confirmed. See `AS-C20_result.md`. Focused suite35passed.
AS-C21 fixed strong-R0 two-I3D-stream screen completed exit0: canonical75.240848,
aware-only30.635838, agnostic-only74.951830, late .1/.9 mix74.662813,
shifted-aware late control74.084778 mean R1. Late loses1.156069pp T2V R1 despite
improving persistent mean ranks−.380435/−.620690(T/V); lead gate FAILS.
Identity tokens/input fusion exact, channel<=1.53e-5 with rank parity;5.32s,
.673GB allocated. All36focused tests pass. See `AS-C21_result.md`.
AS-C22 exact tokenizer parity all7615rows: omissions230train/10dev,1185/38BPE
tokens; no new exact collisions, max content57train/36dev. Three persistent T2V
and two V2T occurrences (three unique pairs) involve affected positives.
AS-C23 full-token restoration75.337187 vs baseline/repetition75.240848 mean R1,
only+.096339pp; persistent V2T mean rank unchanged, lead gate FAILS. No affected
persistent error becomes R1-correct. Other509score columns bitwise unchanged;
new PAD slots excluded equally in full/control.39focused tests pass. See results.
AS-C24 hard-both/A/B mean R1=74.662813/74.470135/74.277457; mean-both/A/B
=33.333333/66.473988/65.317919, all below75.240848 soft baseline. AS-C25
log-mean-exp both/A/B=71.290944/73.314066/73.025048. All nine variants worsen
persistent mean ranks both directions; both scorer cycles FAIL. Soft replay ranks
exact, entropy algebra scaled error1.91e-5, equivalent formulations0rank differences.
Negative similarity-coordinate derivatives are common(.629–.736), not measured
parameter harm.43focused tests pass. See `AS-C24_C25_result.md`.
AS-C26 common5721fit reference: nearest unigram means.485774held/.496031dev,
bigram.315136/.326295; same-prefix support0/1375held,402/519dev. Full7096train
has prefix support519/519dev, NOT proof of clip leakage. AS-C27 same clean
update1000 checkpoint:519-gallery held mean R1 range25.433526–29.094412,
primary27.649326, official dev32.273603. Within dev seen/unseen-prefix R1
33.333333/35.897436T and30.845771/29.914530V: no uniform seen-source advantage.
Both populations weak; original1375held adequacy still FAILED.46tests pass.
See `AS-C26_C27_result.md`. No more scorer/temperature/entropy sweeps.
Post-run local code finding: upstream default freeze_layer_num0 freezes text
embedding/position tables; AS-C20 trained them. README default training example
is How2Sign, NOT verified PH release recipe. Upstream BertAdam versus local
AdamW equivalence now falsified by AS-C28 actual synthetic updates: independent
uncorrected formula error5.55e-17; schedule first factors0/.01, different cosine
progress; caller globally clips before upstream internal clip (no extra change
in synthetic cases). Exactly2text tables/25,336,320parameters frozen by upstream
default, drifted in AS-C20.55decay-predicate mismatches/33,539parameters; logit
scale100.440804vs upstream cap100. These are NOT verified exact PH release recipe.
AS-C29 full1375held rollback mean R1: identity17.490909, position17.418182,
token17.854545,both17.709091; no reset passes. Identity scores exact, other tensors
unchanged. Post-training reset NOT training with frozen tables.49tests pass.
See `AS-C28_C29_result.md` and `AS-C30_result.md`. AS-C30 exact replay then
freeze-only comparison completed, optimizer/lr/decay/clamp/schedule unchanged.
Freeze final held17.381818T/19.127273V, fit94.981818T/95.927273V, held strict
different-input errors1120T/1086V; both tables bitwise unchanged. Positive
relative gain does NOT repair held>=50each adequacy. No more table/optimizer
rescue sweep after AS-C29/C30. Next priority is a materially different text
context-availability audit completed as AS-C31: all7615 reencoded text tensors
match caches; common prefixes>=3BPE among different-input persistent confusers
only6T/3V, registered adverse prefix burden1T/0V. No bidirectional-text model
justified. All nonexact prefix vectors cross original128/tail56or7batch shapes;
32fixed-shape suffix interventions produce EXACT prefix equality and changed
EOT vectors. Those32cases cover11TRAIN/6DEVunique sentence pairs, not independent
replicates. See AS-C31_result.md. AS-C32 fixed3-seed score averaging now gives
77.263969mean,76.878613T/77.649326V, exact repeat controls; recovers1T/4V
persistent queries.88/92T and72/87V persistent queries share a strict confuser
in all3models and cannot be repaired by convex score mixing of those models.
AS-C33 one-model parameter average76.011561mean,76.493256T/75.529865V:
mean+.770713 vsR0,CI[−1.190531,2.666700], V2T−.770713pp. It fails retention
of score-ensemble gain (mean−1.252408,CI[−2.647059,.091747]); do not call it
uniformly better thanR0. All305repeat tensors/cache/scores exact. Retain all
controls and explicit resource contracts, not only the highest mean.
Primary-source screen confirms Model Soups/WiSE-FT cover parameter averaging,
VRF covers the specific failure-bank sample-wise mixing route. Three routes
screened, none are novel surviving candidates. See AS-C32_C33_result.md and
AS-C32_candidate_collision_screen.md. AS-C34fixed3x3crosses completed:
mismatched76.107900,all9=76.589595 vs matched77.263969, pairing signal TRUE;
all3diagonals and6/9pass repeat controls exact. AS-C35shared orthogonal maps
trained on pooled-video/EOTanchors do not repair mismatch (76.011561).
This is neither a semantic bottleneck certificate nor permission to launch
CCA/anchor/layer sweeps. Multi-Way Representation Alignment primary abstract
adds a shared-space/geometry-correction novelty collision warning (full paper
not reviewed). AS-C36 exact deployed visual-input inventory completed:
all7615distinct atFP32/FP16 withvalid masks, zero within/crosssplit collisions;
all15,230source stream files freshly hashed. No equality rechecks needed because
no hash matches; not independent encoder replay. AS-C37 subsequently checks
within-video valid temporal-slot repetition separately: zero repeats in both
precisions,447139valid slots. All input/source hashes matchAS-C36. No conditional
source-equality checks because no repeats. Switch causal layer after these two
bounded negative checks; no near-similarity information ceiling, altered positives,
or sampler revival. See AS-C37_result.md; no active experiment.
Search active, no Proposal8 or global exhaustion.
No ordinary freezing/optimizer change is a novel method or automatic retry.
Do not treat untrained errors as useful supervision, or train a
correction on the same held labels used for confirmation. This is a different
initialization/data regime than R0: require a controlled bridge before any
inference about strong-baseline information sufficiency or candidate gain.
No cohort-teacher, assignment soft targets, new benchmark or dev-label fitting.
All AS-C11 transforms hurt ranking; AS-C12's VALID-token
gradient findings do not justify reopening PMGR/RPCA. The original24-question map is a historical
priority snapshot; measured sections and this current-status header supersede
its old "no/open" cells. All prior closed families remain closed.

## Historical initial-cycle rationale (superseded status text)

Question: are persistent top-10 errors distinguishable in frozen contextual
features, and which deployed scoring assumptions obscure the distinction?
Why selected: highest information-gain score in the map below; existing errors,
checkpoints, and full-gallery controls are available. R0 replay is a prerequisite
for frozen readouts, raw crossed probes, geometry, and order diagnostics alike.

Previous turn classification: progress (diagnostic scaffold and preregistration
created), but NO empirical probe completed. Current process inspection found no
live probe process. No experiment is being silently restarted.

Evidence collected: Proposal 7's historical residual and information-channel
artifacts; selected checkpoint and shared scorer code inspected again. [V] The
Filip scorer's inner softmax includes padded tokens, while outer averaging masks
them. [V] Balanced training averages four directional cross-entropies, not CE on
the mixed score. Neither observation is itself a novel method.

Experiments run: R0 replay; six scorer variants; nine 600-update readout runs;
train hard-pair health check; failed raw subset-batch smoke; batch/precision
diagnostic; corrected three-video raw smoke. Implementation:
`methods/information_probe/`. See measured ledger below.
Findings: exact R0 parity; no learned R1 gain; scorer post-context order invariance;
inference-only masking harms ranking; raw extraction batch shape affects numerical
parity. These are bounded findings, not universal information-loss conclusions.
Falsified hypotheses: this 107,650-parameter/600-step readout beats controls;
padding correction alone improves PH dev; either single directional score channel
alone improves the fixed mixed baseline. Historical exclusions remain binding.
Supported hypotheses: no novel method; frozen readout sufficiency remains unresolved.
Prior-art collisions: listed below as screening leads, not fresh novelty checks.
Closed new branches: the exact AS-C01 readout specification is a negative screen;
not the entire visual-representation or scorer research layer.
Current best candidate: none; evidence and novelty gates unmet.
Next highest-information branch: complete crossed same-I3D spatial versus pooled,
existing-feature and shuffled controls. Other open branches include direct
train-hard-pair readout adequacy, text/augmentation effects, and direction geometry.

## Open research space map

Scores are [I] prioritization judgments, NOT measurements or novelty claims.
Vector = information gain / issue evidence / retrieval relevance / novelty
opportunity / falsifiability / feasibility / distance from closed families,
with maxima 20/20/20/15/10/10/5. Totals guide diagnostics, not method selection.
Prior column gives the nearest already-audited family or an explicitly OPEN
search lead. Exact closest-paper identification is deferred until a mechanism
has support, per the user's targeted-search rule. No claim of absence in SLRet.

| ID; mechanism/question | Layer | Project evidence | Closest prior / collision | Tested locally | Plausibility; novelty uncertainty; feasibility | Cheap falsification | Score; status |
|---|---|---|---|---|---|---|---|
| Q01 Frozen nonlinear recoverability | C/E | [M] AS-C01/C05 negative; train-support limitation measured | CiCo late interaction; ELSC overlap if promoted to local residual method | yes, limited screens | high; high; high | Further sufficiency inference needs a more adequate probe, not repeated head tuning | 20/19/20/8/10/10/4=91; bounded probes negative, sufficiency unresolved |
| Q02 Direction-specific channel geometry | F | [M] AS-C03 independent mixtures underperform | CiCo dual_mix; PMGR grouping CLOSED, not this diagnosis | yes, global mixture | high; high; high | More general geometry remains distinct from failed scalar calibration | 18/18/20/9/10/10/4=89; simple mixture rejected |
| Q03 Candidate-relative discriminability | F/J | [M] AS-C07/C15 cohort headroom survives jitter but consumes other queries;3 candidates rejected | Fixed-bank correction CLOSED, OTTER collision | yes, cohort-only diagnosis | independent-query transfer unresolved | A distinct independent-query gap is needed before another reranker | 19/17/20/9/10/9/3=87; conditional headroom only |
| Q04 Spatial information before pooling | B | [M] AS-C02 30-run crossed screen fails attribution; AS-C44 certifies readout permutation invariance and mixed overlapping pooled axial support | CiCo I3D; active-acquisition proposal 7 rejected | yes, bounded content readout, NOT an explicit arrangement test | Broader relational recovery unresolved; no anatomical interpretation of four cells | No cue-localization claim, positional-head rescue or RF-weighting rebrand from this audit | 20/12/20/10/10/7/4=83; tested probe negative, scope corrected |
| Q05 Padding-mediated score distortion | E | [M] AS-C14 all7 mask interventions lose mean R1; text-pad-only−1.45pp largest single factor | CiCo code; implementation correction not research novelty | yes, full3-factor inference screen | bounded negative; low novelty | Training-consistent policy remains separate, not automatically novel | 17/20/18/1/10/10/5=81; inference mask screen negative |
| Q06 Post-context temporal order blindness | C/E | [M] AS-C06 pre-encoding reversal loses 20.91pp; post-encoding reversal unchanged | CiCo; local-alignment/OT CLOSED | yes, stress test | high; high; high | Valid linguistic minimal contrasts needed beyond permutation stress | 18/14/18/10/10/9/4=83; global pipeline blindness falsified |
| Q07 Pre/post-context information retention | C | [M] AS-C13 bidirectional linear predictability transfers; substitutions hurt R1 | Standard ridge, no novelty claim | yes, limited linear probe | semantic loss unresolved | No information-ceiling inference; no map/teacher sweep | 19/11/19/10/9/8/4=80; bounded retention evidence |
| Q08 Embedding anisotropy suppresses contrasts | C/D | [M] AS-C11 effective rank57; centering/PC1 removal/whitening all hurt | Standard geometry controls, no novelty claim | yes, seven fixed controls | limited negative, not information ceiling | No new metric sweep supported | 15/9/17/5/10/10/4=70; fixed screen negative |
| Q09 Text relational versus lexical sufficiency | D | [M] AS-C16 all61 same-lexicon TRAIN pairs strict/no DEV group; AS-C22/C23 omission restoration +.096pp/no persistent repair; AS-C41 all396 bilingual one-edit TRAIN pairs strict, zero eligible DEV pairs, census independently checked | CiCo CLIP; native-caption preservation and lexical-support families CLOSED | yes, exact-lexicon, omission and bilingual one-edit screens | valid relational labels and held-out support missing; wording edits need not change meaning | No truncation/position/edit-distance rescue; pair existence alone is not a bottleneck | 17/8/18/6/9/9/4=71; registered contrast lead unsupported |
| Q10 Translation collapse beyond exact equality | D | [M] two new exact-collapse classes | Native-caption proposal 7; multilingual priors | partial | medium; low; medium | Native/English similarity disagreement audit, no new encoder | 13/12/15/4/8/9/2=63; diagnostic only |
| Q11 Relevance equivalence versus paired labels | A/I | [M] repeated captions | PMGR CLOSED; official positives immutable | partial | high; high; medium | Stratified confuser/duplicate audit; expert-needed labels | 18/18/17/8/7/8/3=79; open diagnosis |
| Q12 Source-conditioned nuisance confusions | H | [M] AS-C10 matched confusers inconclusive; AS-C26/C27 common fit/model show no uniform seen-prefix advantage across directions | CMCM audit; causal claims require intervention | partial, controlled support comparison | medium; high; medium | Prefixes not verified source identity; no source-only explanation for weak calibration | 16/12/18/9/9/8/4=76; broader nuisance unresolved |
| Q13 Signer nuisance versus linguistic cue | H/B | [M] AS-C10 signer-stratified errors and text-cosine-matched confusers inconclusive; causal attribution unvalidated | Signer/nuisance invariance and selective removal CLOSED in proposal1/5/6 and Proposal7 registry | partial, descriptive confuser and signer slices | broader diagnosis unresolved; closed method family | No generic adversary, nuisance projection or selective-normalization revival; require an open decision consequence before another probe | 15/9/17/0/8/7/0=56; diagnosis only, closure corrected post-C41 |
| Q14 Gradient conflict between directions | G | [M] AS-C18 all6 video-parameter cosines positive; AS-C12 all24 valid-interface cosines positive | RPCA gradient surgery CLOSED | yes, bounded parameter/interface audit | text/optimizer/training stages unresolved | No global opposing-video-direction candidate supported | 18/13/19/8/10/8/2=78; bounded conflict story unsupported |
| Q15 Early selection versus representation drift | G | [M] AS-C09 selected epochs0/-1/146, encoder hybrids do not beat best; oracle-ever recovery is nondeployable | Ordinary early stopping; RPCA not reopened | yes, trajectory and hybrid screen | high; low; high | No oracle-based method claim or checkpoint selection rescue | 17/19/18/3/10/9/4=80; bounded screen negative |
| Q16 Rare contrasts versus easy-negative dominance | G | [M] AS-C18 duplicate-video path projections heterogeneous; no actual update measured | PMGR/mining CLOSED | yes, bounded pathway attribution | query-loss attribution and harm unresolved | Do not equate pathway projection with harmful parameter pressure | 15/9/18/6/9/8/1=66; concentration measured, no miner |
| Q17 Simultaneity represented as relations | B/C/E | [M] AS-C44 invariance/support certificate; post-C44 all14711 permitted CSV rows lack populated start/end; [V] release README excludes manual ECCV mouthing sequences from ALL PH2014T sets; [U] actual joint-contrast burden | Koller synchronization, multi-channel Transformer and STMC prior art; RCLI/global composition remain CLOSED | architecture yes, linguistic intervention no; direct published-mouthing join ruled out by release contract | Documented exclusion, not independent footage-overlap proof; no isolated articulators in cached bins | Stop direct archive-join route, switch question; no pseudo-label, filename, generic stream/position/synchronization rescue | 18/5/19/4/2/4/4=56; specific resource route closed, science unresolved |
| Q18 Candidate multiplicity and direction asymmetry | A/F | [M] repeats; PH paired gallery | PMGR population-risk CLOSED | partial | high; low; high | Duplicate-aware descriptive slices, unchanged evaluator | 13/18/17/3/9/10/1=71; diagnosis only |
| Q19 Temporal sampling aliasing | B | [M] AS-C17 fixed nearby-view mean−.48pp;475 videos have1 effective view across seeds | Sampling-consistency and partial OT CLOSED | yes, low-diversity view control | broader aliasing unresolved | No sampler/seed tuning to rescue this screen; no global acquisition rejection | 16/9/18/8/10/7/2=70; bounded negative |
| Q20 Feature geometry versus score calibration | E/F | [M] AS-C24 hard/mean endpoints and AS-C25 entropy removal all lose mean R1 and persistent ranks | Proposal 7 nuisance correction rejected; generic pooling not novel | yes, fixed endpoints and algebra | broader geometry unresolved | No temperature/entropy/padding rescue sweep after two scorer failures | 16/12/19/6/10/10/3=76; tested mechanisms unsupported |
| Q21 Augmentation destroys rank-critical wording | D/G | [M] AS-C39exacttrainingreplay; AS-C40six matched260update runs,OFF−ON−.385/+3.661/−1.541pp,0/3allgatespass,allOFFbelowinit | RPCA/generation CLOSED; generic augmentation removal not novelty | yes, frozen/gradient and actual learning intervention | registered fixed-horizon harm lead unsupported; other horizons unresolved | No strength/schedule/precision/seed rescue; choose a different causal layer | 17/10/18/3/10/9/4=71; controlled diagnostic negative |
| Q22 Independent scalar score expressivity | E/J | [M] AS-C15 dual additive offsets support assignment non-strictly; cohort-dependent | Assignment duality, not novel inference | yes, bounded algebra/data certificate | nonadditive necessity unsupported | Do not equate joint headroom with a scalar-score expressivity ceiling | 19/10/20/12/10/8/4=83; strong inference not established |
| Q23 Genuine unavailable information / ambiguity | I | [M] persistent errors, not irreducibility | Annotation-limits literature OPEN | partial | plausible; high; low | Audit after successful powered readouts; never infer from one failed probe | 18/10/17/9/6/5/5=70; unresolved |
| Q24 Cross-dataset failure-signature transfer | H/J | [V] H2 provenance and grouping caveats | CiCo/UPRet; no new benchmark | partial | high; high; medium | Measure surviving signature on locked CSL/H2 dev | 18/10/20/9/9/6/5=77; conditional confirmation |
| Q25 Clean residual model calibration and recipe fidelity | G | [M] AS-C30 exactly reproduces AS-C20; freeze-only held18.254545 vs17.490909 but original>=50each adequacy fails | Ordinary optimizer/freezing corrections, not novelty; no OTTER revival | yes, isolated freeze comparison | bounded positive relative effect, inadequate residual model | No table/optimizer rescue sweep after AS-C29/C30; other recipe differences unresolved | 20/20/19/0/10/10/5=84; tested factor insufficient, not candidate |
| Q26 Text-token right-context availability | D | [M] AS-C31 broad shared-prefix burden1/92T,0/87V; all32fixed-shape suffix interventions preserve prefixes exactly while EOT changes | CiCo/CLIP code; no absence-of-prior claim | yes, bounded signature and numerical check | broad prefix explanation unsupported; other context limits unresolved | No bidirectional-text model or prefix correction justified from this screen | 18/10/18/8/10/10/4=78; tested signature rejected |
| Q27 Deployable cross-checkpoint complementarity | F/J | [M] AS-C32ensemble77.263969; AS-C34mismatched76.107900,CIbelowmatched; AS-C35shared orthogonal map76.011561 does not repair despite transferable anchor fit | Model Soups/WiSE-FT; VRF failure-bank mixer; Multi-Way Representation Alignment abstract collision warning | yes, averaging/crossing/fixed coordinate control | supported pairing sensitivity, no novel bottleneck yet | No mixture/anchor/layer/alignment rescue sweep; keep explicit3-model baseline | 18/20/19/0/10/10/5=82; stronger control, candidate mechanism unresolved |
| Q28 Exact deployed visual input distinguishability | A/B | [M] AS-C36 all7615whole inputs distinct; AS-C37 zero within-clip valid-slot repeats atFP32/FP16,447139slots; allsource/input hashes match | Information-availability diagnosis, not changed labels/benchmark | yes, whole-input and within-clip inventories | both exact-collapse explanations unsupported, no semantic ceiling | Switch layer after two negatives; no similarity/precision/sampler rescue | 18/8/18/0/10/10/5=69; bounded exact-collapse branch negative |
| Q29 Independent-query numerical batching stability | J | [M] AS-C38 tail padding changes5single V2Ttie ranks/oneR1, no strict preference sets; ALLensemble ranks unchanged,+2.023121ppgain survives | Computational validity control, not novel batching/precision/evaluator method | yes, original and padded repeated encodings/scores exact | bounded tie sensitivity, no semantic defect or candidate | No batch/precision/sorter rescue sweep; preserve official evaluator and3-model control | 17/16/20/0/10/10/5=78; fixed test complete, ensemble robust to it |
| Q30 Adequate TRAIN-only residual learning at larger fixed budget | G, prerequisite for C/E diagnosis | [M] AS-C42 completed8800updates heldmean22.472727 vsAS-C20short17.490909; bothhelddirections<50; all8metrics/finalscore replay exact | Ordinary learning-budget calibration, not novelty or historical upstream PH reproduction | yes, completedextendedbudget andvalidation; interruptedprefixreplayed | Registeredbudgetfailsadequacy, no calibratedresidualsupervision; Q01unresolved | No duration/schedule/optimizer/freeze/seed rescue; move to materiallydifferent mechanism | 20/10/19/0/10/10/5=74; bounded prerequisite lead insufficient |
| Q31 External discourse context in sentence clips | A/I | [M] Post-C42 metadata census: all7615temporal files sentence-local; DEV117same-split/292TRAIN numeric predecessors, independently counted but not verified continuity | Sincan2023/Jang2025 preceding-context translation; Yin2021/Baltatzis2026 coreference; RPCA context-protection CLOSED | pre-candidate metadata and primary-literature boundary audit only | No PH linguistic or retrieval signature; How2Sign human result not a PH ceiling | No inferred-neighbor caption input, context/protection or source-calibration rescue; seek a different testable open mechanism | 17/6/16/2/4/8/3=56; current route not promoted, wider discourse science unresolved |
| Q32 Independent-query gallery-neighbor support | J/E | [M] AS-C43fixedk5 smooth/sharp real/shuffle,3checkpoints+ensemble:0/3gates each; ensemble−6.551/−.578pp vsstrongest77.263969;32scorematrices independentlyreconstructed | Iscen2017diffusion/Zhang2020GNNre-ranking; graphfiltering known, not candidate-onlyconstant offsets | yes, fixedgallery operators withsingle-query parity andshuffledtopologycontrol | Bothfixedoperators fail, broadergallerystructure unresolved; no scalar-expressivity ceiling | No graphsize/strength/pooling/layer/temperature/learnedgraph rescue; switchcausallayer | 18/8/20/0/10/10/4=70; registeredleadunsupported |

Additional map entries from the latest cycles:

| ID; mechanism/question | Layer | Evidence and decision | Next gate |
|---|---|---|---|
| Q33 Exact-gloss persistent confusers | A/I | AS-C45:1/92T2V,2/87V2T acrossallseeds; bothbelow10% material-burden gate; census/scalarvalidation exact | Signature unsupported; no fuzzy-gloss/mining/equivalence rescue |
| Q34 UPRet probability-score reduction fidelity | G/E | FinalEq18sum/source max discrepancy confirmed; realTRAIN B32 shared-noise screen: amplitude-controlled median gradientchange .174891%visual/.220684%text atpartialstep767, both<1%gate; exactreplay | Deprioritize long reduction campaign; no temperature/weight/optimizer rescue. Limited gradient screen, not universal closure or retrieval result. Change causal layer. |
| Q35 CMCM covariance gradient fidelity | B/G prerequisite | Composed validPSD finite differences confirm epsilon normalization mismatch; isolated denominator control restores exact-forward gradient to1.12e−16; no trained exposure measurement | Repair prerequisite only, not novelmethod. No surrogate CiCo activation-distribution substitution, epsilon sweep or replacementpipeline campaign. |
| Q36 CiCo pseudo-clip nonlocal grouping exposure | A/B prerequisite | Known grouping behavior, new complete localTRAIN index census:68/9854training segments withgaps, expected any-gap crop0.615417%<10%gate; frame/interval/sampler checks and replay exact | Deprioritize specific repair route; no semantic-noise or retrieval-harm claim, OCEM staysclosed. No NMS/crop/teacher/fusion rescue; change question. |
| Q37 Subword identity / fingerspelling attribution | D/A prerequisite | How2Sign FS convention verified, current human-label join unavailable/unverified; owner release statement and current site checked. FSS-Net/character-level SLT prior art. CiCo inverse byte/symbol maps refute a source-only spelling-erasure inference | No tokenizer/character/teacher/crop/extra-stream candidate. Existing-label attribution route not feasible now; no PH/How2Sign failure prevalence measured, wider mechanism unresolved. |

Closed mechanisms: ELSC, DIVE-SLR, PLEL, OCEM, SSSC, sampling-consistent partial
alignment, PMGR, RPCA, historical R1–R5, and Proposal 7's rejected native-caption,
nuisance prior-column, active-acquisition, and conditional-density candidates.
Diagnostic use of an overlap is not permission to claim it as a new method.

## Completion / exhaustion ledger

The map currently contains Q01–Q37. Earlier reference to "24 branches" was stale.
The all37 decision/scope audit is now in
evidence/autonomous_search/Q01_Q37_feasibility_audit.md, with a hashed explicit-link
inventory. It distinguishes failed specifications, collisions, missing resources
and unresolved formulation/attribution gaps. It does NOT establish all remaining
high-value interventions infeasible; no terminal barrier or exhaustion claim.
Q02/Q22's analytical follow-up rejects two overbroad necessity arguments, not
the implemented-function or learning questions. No new method meets GO.
Before proposal 8: measured bottleneck; >=3 distinct derived candidates; targeted
primary-source collision search; five inline adversarial perspectives; minimum
pilot; B0/B1/B2/B3 controls; fixed three-seed/cluster-bootstrap gates; second-data
signature or dataset-independent proof and feasible confirmation plan.

Evidence directory: `evidence/autonomous_search/`. Tensor caches remain in
`artifacts/proposal7/phase2/` to avoid duplicating large files.

## Measured experiment ledger

All runs use git `a5fe287536db55b44ce519050a23d389f27d9c2e` plus recorded untracked
diagnostic source hashes. No commit/push. Run JSON files contain exact checkpoint
and manifest hashes, configuration, seeds, timing, hardware, and per-step records.

| Experiment | [M] result | Consequence |
|---|---|---|
| AS-C01-R0 | Exact saved-score equality; zero rank differences; 74.1811/76.3006 T2V/V2T R1; all 7,096 train and 519 dev examples cached | Reliable common frozen input, not a new method |
| AS-C01-SCORER legacy reimplementation | max channel difference 1.53e-5; no rank differences | Diagnostic kernel validated against upstream |
| AS-C01-SCORER masked | 72.6397/74.7592; mean -1.5414 pp | Reject inference-only masking as an immediate gain; training-consistent correction untested |
| AS-C01-SCORER reverse contextual tokens | No rank changes, floating reduction difference <1e-5 | Scorer invariant after encoding; pre-encoding order information remains untested |
| AS-C01-SCORER pooled | 47.5915/41.8112 | Discarding token matching severely harms this fixed score; not a learned-control result |
| AS-C01-SCORER individual directional channels | mean R1 70.9056 and 71.5800 vs 75.2408 mixed | Neither channel alone is an immediate improvement |
| AS-C01-R1 pooled / interaction / zero x seeds 42/1337/2026 | 600 updates each; all nine selectors retain step 0; all mean R1 75.2408; learned gain 0 | Fixed screen fails; no positive conditional bootstrap effect |
| AS-C01-R1 aggregate | strongest-control delta 0; 10,000-draw source-prefix cluster CI [0,0]; 0/3 seeds pass | Not a method GO; a degenerate CI because selected rankings are identical |
| AS-C01-HEALTH | train hard-pair strict accuracy 96.1809 T2V / 95.4622 V2T; 271/322 nonpositive margins | Training errors remain; no proof of gradient starvation or information absence |
| AS-C02-RAW-SMOKE | failed strict pooled parity, relative drift .0002663 | Failure recorded, no raw training launched |
| AS-C02-RAW-PARITY | full historical batch32 exact; batch8 drift; TF32-off drift .0017273 | Preserve historical batch composition; do not relax numerical gate |
| AS-C02-RAW-SMOKE-B32 | all three dev videos exact pooled parity | Full extraction permitted, unchanged numerical and retrieval gates |

Training-screen limits: one fixed backbone across probe seeds; clean text without
dynamic swaps for all arms; four learned interaction summaries plus pooled pair
features; no temporal-order-aware readout. These limitations prevent concluding
that the representation is insufficient. The zero-input control is a capacity
control, not a substitute for a trained shuffled-information raw control.
Actual timings are reported; compute equivalence is not inferred from parameter
count alone. Statistical caveats and all 11 fallacy checks are in
`evidence/autonomous_search/AS-C01-R1_summary.json`.

## Active execution handoff

Full extraction command:
`PYTHONPATH=shared:. timeout 7200 /home/haipd/miniconda3/bin/python -m methods.information_probe.cache_raw --splits dev train`

Unified session: `39041`. Authoritative progress:
`evidence/autonomous_search/AS-C02-RAW-CACHE-B32_run.json`.
Re-poll the session/PID before deciding whether it has stopped. Never restart from
this note alone. Complete caches will be `artifacts/proposal7/phase2/raw_dev.pt`
and `raw_train.pt`. The smoke cache is incomplete and must not enter training.
Only acquisition is currently running; R2/R3 retrieval training is NOT complete.

Latest verified check: dev complete (519/519, zero pooled discrepancy), train
progressing; process PID 3027869 was confirmed live. This count is a checkpoint,
not a live status source. Read the run JSON/session again on continuation.
R2/R3 runner and same-capacity information controls are implemented and six unit
tests pass. Protocol and the pending training command are in
`evidence/autonomous_search/AS-C02_crossed_screen_protocol.md`.
The 24-case deterministic stratified error audit is saved as
`AS-C01-ERROR-AUDIT.json`; sign-language validity remains expert-unverified.
The ten-test combined diagnostic/historical-protocol check also passed.

## AS-C03: train-only direction calibration, completed while acquisition runs

Previous goal turn classification: **progress** (new measurements, nine completed
controlled runs, raw acquisition implementation and live execution). On resumption,
PID 3027869 was confirmed live; it was not restarted.

Question Q02: does independent fitting of existing score-channel mixtures improve
the two retrieval directions? Fit a bounded scalar mixture on all 7,096 fixed
train confusers per direction, minimizing pair-logistic loss. Dev is evaluation
only. Shared-fit and train-shuffled-direction controls use the same train pairs.
This is a deterministic calibration diagnostic—not a novel candidate or three-seed
method campaign. Full directional score matrices, metrics, and per-query deltas
are retained in `AS-C03-*` artifacts.

| Fit | T2V channel-0 weight | V2T channel-0 weight | T2V R1 | V2T R1 | Mean R1 |
|---|---:|---:|---:|---:|---:|
| R0 | .5 | .5 | 74.1811 | 76.3006 | 75.2408 |
| Shared train fit | .439626 | .439626 | 73.7958 | 75.5299 | 74.6628 |
| Direction-specific train fit | .364433 | .559422 | 73.2177 | 75.9152 | 74.5665 |
| Shuffled-direction train fit | .447610 | .431804 | 73.7958 | 75.3372 | 74.5665 |

[M] Direction-specific training optima differ, but the fitted weights reduce
full-gallery dev mean R1 by .6744 pp and do not outperform the shuffled-direction
control. V2T persistent mean rank worsens by .2644. [I] This rejects simple
global directional channel calibration as a useful intervention here. It does
not establish that shared geometry is fundamentally adequate, nor justify a new
asymmetric architecture. No targeted literature search or candidate generation
is warranted by this result alone.

AS-C02 remains active. Margin extraction briefly overlapped its GPU work;
reported wall times are NOT isolated compute-benchmark evidence.

## Dependent campaign execution and fairness refinement

Before any raw retrieval training, the existing-feature control was strengthened
to retain 32 uniformly sampled valid contextual tokens, not only a temporal mean.
This removes a preventable token-access confound. Nine diagnostic unit tests now
pass, including paired-versus-matrix channel parity, convex-fit endpoint behavior,
and preservation of token distinctions in this existing-feature control.

The registered R2/R3 campaign has a live dependent runner:
`PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.continue_crossed --extraction-pid 3027869`

Unified session `10903`; authoritative state:
`evidence/autonomous_search/AS-C02-CAMPAIGN_run.json`.
It follows the exact currently live extractor PID/start-time/command identity,
requires terminal `completed` extraction, then executes the registered 30-run
matrix and `summarize_raw.py` once. It never restarts extraction or retries a
failed stage. Both dependent commands have 1,800-second timeouts. Stage stdout/
stderr is captured in `AS-C02-CAMPAIGN_stage0.log` and `stage1.log` when launched.
The training loader independently checks complete IDs, manifest/checkpoint hashes,
and all raw pooled-parity results. Do not also launch the training command manually
while this runner is live. This is scheduled execution, not a claim of completed
raw retrieval experiments. Check both live processes on the next continuation.

## 2026-09-15 completion of AS-C02 and updated interpretation

The extraction and dependent campaign are now **completed**, confirmed by terminal
run records, 30 completed run files, cache hashes, 18,000 training updates and the
generated crossed summary. Previous process handles are no longer live; do NOT
restart them. Extraction processed all 519 dev and 7,096 train videos, with exact
last-reported pooled parity and every row passing the strict gate. Raw extraction
took about 4,089 seconds; the 30 training runs sum to 237.08 seconds, not including
extraction. Exact per-run resource records remain authoritative.

| Regime | Spatial mean R1 | Strongest matched control | Control mean R1 | Difference pp | Source-cluster 95% CI | Gate |
|---|---:|---|---:|---:|---|---|
| R2 | 75.433526 | train-shuffled spatial | 75.401413 | +.032113 | [-.326174, .385356] | FAIL, 0/3 seeds |
| R3 | 75.240848 | existing-token processing | 75.272961 | -.032113 | [-.221800, .159642] | FAIL, 0/3 seeds |

The small R2 change versus R0 is not attributable to real spatial information.
No raw cue localization or new architecture is justified by this result.
These are limited negative screens, not formal upper bounds or a global NO-GO.

## AS-C04: a measured training-support limitation

Exact deployed tokenization gives 6,825 distinct train inputs among 7,096 rows;
51 repeated-input classes contain 322 rows. Every one of the 271 negative T2V
train hard-pair margins has an identical query input to its confuser. The strict
T2V train hard-pair accuracy 96.180947% equals the one-top1-per-input
identifiability bound. Dev is different: 507 distinct inputs among 519 rows,
only 17 repeated-input rows, but 92/87 persistent errors.

V2T: 278 near ties and 44 tiny negative margins. Of these, 321 involve identical
deployed inputs. The single different-input pair is index5581 versus2666, two
English forecast-introduction captions differing by “the”/punctuation; the native
captions refer to different months. Its margin is only about -.000612. Signed
semantic equivalence remains expert-unverified.

An additional numerical inspection found that identical greeting text encoded
in the final short train batch differs from earlier full-batch encodings by up
to .00390625 in contextual features. Many tiny V2T negative margins point to
that final-batch greeting. This is a cache/batch-precision diagnostic, not a
linguistic distinction or method gain; do not quietly alter historical R0 caches.

The first-30-token shortcut in Proposal7's collision script is not the deployed
long-caption tokenizer (which samples content tokens uniformly). Rechecking exact
IDs changed no duplicate-membership counts in train/dev, so the historical
collision counts survive; the shortcut must not be reused for new input audits.

[I] The residual readout screens had essentially no distinguishable train ranking
failures resembling the dev population. Their negative results cannot determine
representation sufficiency. Classification remains **E / diagnostic identification
incomplete**, not D / semantic irreducibility. This finding does not reopen PMGR.

## Active AS-C05 execution

Registered protocol: `evidence/autonomous_search/AS-C05_signal_protocol.md`.
Same readout and inference, with baseline coefficient in training loss crossed
between0 and1, each at 2,400 updates; matched modes/seeds. The coefficient0 arm
tests whether a readout can learn without a saturated frozen-baseline term. This
intentional loss/inference mismatch is a diagnostic—not a proposed method.

Currently launched: coefficient0, unified session `48150`.
Command: `PYTHONPATH=shared:. timeout 1800 /home/haipd/miniconda3/bin/python -m methods.information_probe.train_readout --modes pooled interaction zero --seeds 42 1337 2026 --steps 2400 --experiment-prefix AS-C05-loss0 --loss-baseline-weight 0`.
Coefficient1 matched 2,400-update runs remain to be launched after it terminates.
Check process/run records before restarting or making claims about outcomes.

## AS-C05 completed: stronger learning signal did not yield retrieval gains

Both coefficient conditions completed all nine runs (18 total, 43,200 updates).
No jobs from AS-C05 remain live. `AS-C05-SUMMARY.json` includes selected and last
metrics, all seed results, fixed-train-pair discrimination and cluster intervals.

Coefficient0 learned nontrivial discrimination: informative standalone heads
reach roughly 77–83% on different-input train confusers, versus the zero-input
control's tied scores. Nevertheless, all coefficient0 dev selectors retain
initialization; final informative-head mean R1 is only about 66.76–69.17%.
The matched coefficient1 runs have small selected gains, at most .193pp, below
the fixed .5pp requirement and not evidence of a new mechanism.

Coefficient0 minus coefficient1, seed-averaged paired differences:

- Pooled: -.096339pp, source-cluster95% CI [-.626579, .451322].
- Interaction: -.128452pp, CI [-.634921, .392927].
- Zero input: 0, identical ranks.

[I] Removing the saturated baseline term lets the limited readout learn, but is
not a sufficient remedy for dev failures in this diagnostic. The deliberate
loss/inference mismatch and remaining standalone train errors prevent claiming
that all scoring/optimization mechanisms are exhausted. Do not keep tuning the
same residual family; diversify to candidate-set structure and other layers.
Fixed-confuser accuracy can exceed a full-gallery identifiability bound after
training because the strongest competitor may change; it is not full-gallery
train recall. That distinction is explicit in the summary's limitations.

## AS-C06 completed: localize temporal sensitivity within the existing pipeline

Same checkpoint, same text features, all519 dev candidates, no learning or dev
selection. Identity replay is exact. Only input-window ordering/content changes.

| Perturbation | Mean R1 | Difference from R0 |
|---|---:|---:|
| Identity | 75.240848 | 0 |
| Reverse input windows before encoder | 54.335260 | -20.905588 |
| Deterministically shuffle input windows | 66.281310 | -8.959538 |
| Repeat within-video mean feature | 7.418112 | -67.822736 |

After undoing the permutation to compare corresponding input windows, contextual
token cosine distance remains .29735 for reversal and .23137 for shuffling.
Contrast with AS-C01's post-context permutation: no changed ranks. [M] The encoder
already responds strongly to global input arrangement and feature heterogeneity.
[I] The full pipeline is not globally order-blind. A post-context invariant scorer
does not establish that the representation omitted temporal information.
These are stress interventions, not expert-validated linguistic minimal pairs;
they do not prove that the observed sensitivity is specifically grammatical.

Evidence: `AS-C06-TEMPORAL_run.json`, full scores and per-query variant metrics.
The padding/content-preservation test passed. No AS-C06 process remains live.

## Next research decision

Q03/Q22 candidate-set structure is a high-value untested layer after these bounded
representation/optimization negatives. First perform label-free structural
diagnostics and distinguish query-independent ranking from cohort-dependent
assignment. Any use of the complete query cohort or one-to-one capacity must be
marked transductive/assumption-dependent, not passed off as a deployable method.
Do not reopen Proposal7's prior-column correction or PMGR under a new name.
No new method, proposal8, or global research-barrier conclusion is supported yet.

## AS-C07 completed: conditional headroom from cohort-coupled inference

Label-free maximum-score assignment uses the SAME saved pair scores, but adds
complete-query-cohort access and one-to-one capacity. It does not use paired
labels in its solver. All519 candidates remain in each reported ranking.
This changes inference assumptions/resources: it is a diagnostic, not a standard
independent-query method result. It tests independent argmax, not a theorem that
an arbitrary scalar pair score cannot represent correct retrieval.

| Condition | Seed42 mean R1 | Seed1337 | Seed2026 | Seed-averaged gain vs respective R0 | Source-cluster95% CI |
|---|---:|---:|---:|---:|---|
| Full query cohort | 83.622351 | 83.044316 | 81.695568 | +7.803468pp | [5.867138, 9.829060] |
| Half-query cohorts | 77.263969 | 77.552987 | 76.396917 | +2.087347pp | [.991635, 3.206413] |
| Exact independent reference | 75.240848 | 74.759152 | 74.951830 | 0 | [0,0] |

`AS-C07-COHORT.json` preserves the original diagnostic. V2 adds an exact untouched
independent reference because forced single-query argmax promotion changes legacy
tie behavior (a small -.096339pp mean effect), even though it adds no competition.
Use `AS-C07-COHORT-V2.json` for the clarified comparison. No thresholds were changed.
Full/half-cohort numbers reproduced unchanged in V2. Per-seed full-gallery ranks,
margins and persistent subsets are saved in the corresponding metric artifacts.

[M] Cohort structure can substantially change correctness under this strong
capacity assumption. [I] This is conditional inference headroom, not proof that
the same gain is available to an isolated query, that one-to-one semantic relevance
is valid, or that a novel method exists. No new raw information was introduced.

Three distinct follow-up mechanisms and five inline adversarial perspectives are
documented in `AS-C07_candidate_screen.md`:

1. Fixed train-bank marginal assignment: rejected. Exact forced-edge value is
   s(q,j)+M(bank,V\{j}), a candidate-only additive potential. This collapses into
   Proposal7's closed correction family, rather than a new conditional interaction.
2. Out-of-fold cohort-teacher distillation into an independent scorer: OPEN for
   targeted novelty and clean-initialization screening, **not selected or GO**.
   PH-fitted release initialization cannot be treated as clean held-out provenance.
3. Standalone permutation-structured training: rejected as underidentified for
   ordinary retrieval. Row/column potentials cancel from every complete assignment
   loss but can change independent rankings arbitrarily.

`AS-C07-ALGEBRA.json` verifies candidate1's reduction (max error4.44e-16) and
candidate3's invariance: a toy score transformation leaves assignment/loss
unchanged while independent R1 changes100%→25% in both directions. These kill the
standalone formulations; they do not license renamed variants or prove all
structured mechanisms impossible.

Next action: targeted primary-source collision search for query-bank normalization,
transductive-to-inductive/listwise distillation, and assignment-based contrastive
training, followed by a provenance audit if the remaining mechanism is not already
covered. Do not start expensive teacher-fold training before this screen.
All experiments currently listed above have terminated; no live probe job remains.

## AS-C07 targeted novelty screen completed

`AS-C07_collision_screen.md` records primary-source reading scope, search strings,
access failures and five included papers. Closest collision for B is OTTER
(ICLR2022; detailed reading of arXiv2023 revision): teacher similarities feed
balanced transport, then matching probabilities supervise contrastive learning.
The current B adds held-out folds and a different application/teacher but no
materially distinct causal operator. REJECT this specification, not all future
structured retrieval. A/B/C all fail the current selection screen. No fold
training, Proposal8, or global research-barrier conclusion follows.

## AS-C08 completed: exact deployed augmentation is a limited perturbation

Protocol: `AS-C08_protocol.md`. Report: `AS-C08-TEXT-AUG_run.json`.
Exact clean token parity on all7096 train/519 dev and exact clean dev scores.
Three augmentation-only seeds;10.08s total,2.33GB peak allocated. No updates.
The actual scorer uses clean text for channel A, augmented text for B; both
channels' matrices enter both retrieval-direction CEs under balanced training.
The mixed-score stress is not the four-CE training objective.

| Augmentation seed | Dev both-aug mean R1 | Dev A-clean/B-aug mean R1 | Train different-clean-input nonpositive fixed margins, T2V/V2T, A-clean/B-aug |
|---|---:|---:|---|
|42|74.373796|74.470135|0/1|
|1337|73.988439|74.951830|0/0|
|2026|74.951830|75.144509|0/0|

Clean reference75.240848. Interpretation: exact augmentation does not create a
large distinguishable fixed-confuser train-error population at this checkpoint.
Some repeated-clean-caption ties are broken by independently sampled swaps;
that is NOT evidence of disambiguating signed meaning. No inference that removing
augmentation improves training. Full objective/gradient effects remain open.

## AS-C09 completed: historical dynamics and crossed endpoints

`AS-C09-TRAJECTORY.json` covers200/201/201 dev evaluations. Selection epochs
are0,-1,146 for seeds42,1337,2026; epoch0 is13updates, -1 is release initialization.
This rejects a blanket "all models peak early" account. Epoch-level train losses
are dynamic-minibatch/augmentation averages, not fixed full-gallery loss; seed42
first/last means.037846/.038299 do not support a simple monotonic-overfit story.

Ordering check: all519 selected ranks in both directions exactly match each
saved evaluation; historical evaluator revisions use sequential first-occurrence
ID order. Across the trajectory,104–116 selected errors per direction/seed become
correct at least once. Only17–23 queries are always wrong. This label-dependent
ever-correct envelope is NOT a deployable method or available improvement; 200
looks amplify it. Median selected-error correctness occurs in only5–7% of epochs.
Only best/last checkpoints remain; no invented intermediate representation data.

`AS-C09-CROSS_run.json` uses four preregistered endpoint combinations per seed.
All three best/best score replays are bit exact. Full-gallery mean R1:

| Video / text state | Seed42 | Seed1337 | Seed2026 |
|---|---:|---:|---:|
|best/best|75.240848|74.759152|74.951830|
|best/last|69.075145|69.171484|74.759152|
|last/best|74.181118|74.566474|74.277457|
|last/last|74.181118|73.025048|74.373796|

No hybrid beats its selected baseline. Seed42/1337 valid-token mean cosine drift
is about.234–.238 video and.199–.207 text; seed2026 late endpoints differ by
.000143/.000097. The asymmetry may reflect coordinate compatibility, not causal
harm or information loss in one tower. No freeze-tower, soup, distillation or
RPCA variant is promoted as novelty. Both script development errors (st_size
property, absent epoch-1 training loss) are preserved in execution notes.

## AS-C10 completed: source/signer confuser association

`AS-C10-NUISANCE.json`; same519 dev pairs and three saved baseline matrices,
native metadata and a crude20-neighbor pooled-text-cosine comparator. No fitting.
Persistent T2V strongest confusers are same signer16–19/92; V2T13–17/87.
Excess vs text-cosine-matched comparator is+.22 to+3.64pp T2V, but−2.82 to+2.30pp
V2T. Same-source counts are0–1/92 T2V and1–2/87 V2T. This does not establish a
large stable same-group confusion signature. It is not a proof nuisance is absent:
covariate gaps average.058–.062 T2V/.043 V2T, content matching is imperfect,
metadata are not independent linguistic annotations, and no causal intervention
or uncertainty inference was performed. Do not design a generic adversary from it.

Validation:18 tests passed after AS-C08; all subsequent experiment parity/order
assertions passed; compileall and git diff --check passed. No test dataset read,
SEDS asset, external upload of private corpus, commit, or destructive action.

## AS-C11 completed: anisotropy exists, fixed removal controls hurt

Protocol `AS-C11_protocol.md`; report `AS-C11-GEOMETRY_run.json`; saved fitted
transforms `artifacts/proposal7/phase2/AS-C11-transforms.pt`. All7096 train rows,
sequence-balanced valid-token moments then equal modality weighting, no labels.
Shared covariance participation ratio32.05 and entropy effective rank57.02/512;
PC1 accounts9.78% variance, first10 components42.69%. These are geometry
descriptors, not proof rank-critical information is suppressed.

| Fixed transform | Mean full-gallery dev R1 | Delta vs identity |
|---|---:|---:|
|identity|75.240848|0|
|shared center|71.868979|−3.371869pp|
|separate center|72.254335|−2.986513pp|
|shared centered PC1 removal|68.978805|−6.262042pp|
|shared centered random1 removal|71.868979|−3.371869pp|
|shared ridge whitening|58.092486|−17.148362pp|
|random-orientation equal-spectrum whitening|71.676301|−3.564547pp|

Identity channel discrepancy1.5259e-5 within prior kernel tolerance2e-5; all
ranks unchanged. Statistics fit CPUfloat64; scorer CUDAfloat32. Runtime3.72s,
peak0.400GB allocated. All variants disclosed; no dev selector, three-trained-seed
claim or new method. Padding transformed along with real slots; no separate
masking intervention. Negative controls rule out these specified corrections,
not arbitrary learned geometry or representational information sufficiency.

## AS-C12 completed: gradient concentration, not overall directional conflict

Protocol `AS-C12_protocol.md`; raw24 conditions and batch indices in
`AS-C12-GRADIENT_run.json`; aggregate/11-fallacy check in `AS-C11-C12-SUMMARY.json`.
Three sampling/augmentation seeds, four512-example batches each, clean and exact
deployed augmentation. No optimization, dev data, hard-negative mining or
parameter updates. Exact clean text re-encoding parity passed.

IMPORTANT scope: summary cosine/norms exclude video CLS/padding via video_mask;
these are gradients in separate per-example VALID contextual-video-token
coordinates. They are NOT shared-encoder parameter gradients. Jacobians may
cancel, align, or redistribute them. Invalid-token gradient pressure is unmeasured.

| Quantity, mean across12 batches per condition | Clean | Deployed augmentation |
|---|---:|---:|
|Global directional cosine|.07916|.22248|
|Fraction examples with opposing local cosine|1.6439%|1.2370%|
|Within-batch duplicate clean-input fraction|2.4902%|2.4902%|
|Duplicate-row share of combined gradient squared norm|83.5098%|56.5299%|
|Top10 rows' share of combined gradient squared norm|86.9288%|85.1230%|

All24 global cosines positive. Clean duplicate mass range73.25–97.38%; augmented
13.72–92.46%, so do not claim uniform concentration in every augmented batch.
This strengthens AS-C04's train-support explanation and weakens a simple
opposing-direction story at the measured interface. It does not establish harmful
learning, rare linguistic contrast neglect, or a new optimization mechanism.
Closed PMGR covers population/duplicate-risk remixes; closed RPCA includes
gradient-surgery/protected-update remixes. No renamed candidate or training
campaign is justified from this result alone. Parameter-space and earlier-training
behavior remain unresolved, not empirical proof of irreducibility.

AS-C12 runtime15.50s,26.68GB peak allocated. Final focused validation21 tests
passed in1.03s, compileall/git diff --check passed. Both experiment sessions
returned exit0 and no live probe remains. Goal remains active; no Proposal8.

### AS-C13 — transferable linear token predictability, no retrieval gain

Protocol `AS-C13_protocol.md` was written before execution. Run
`AS-C13-RETENTION_run.json`, all12 reconstruction cells and five full-gallery
metric/score pairs are retained; compact accounting and11/11 fallacy checks in
`AS-C13-SUMMARY.json`. Implementation `retention_probe.py`, two algebra/masking
tests and `summarize_retention.py`. No test, external service, new backbone or
changed retrieval labels. Same pinned R0/cache/manifest hashes as earlier probes.

Fit7096 train sequences/417323 valid aligned windows with equal sequence mass.
Normalize1024D pre-context and512D post-context tokens; skip CLS/padding in fit.
Fixed affine ridge `.01*trace(Cxx)/d`, CPUdouble, no sweep/dev selection. Seed42
global valid-token permutation provides a same-size correspondence-null fit.
Input features loaded through the same canonical shared dataset, IDs and valid
masks asserted. Inputs kept in RAM; only small map artifacts persisted.

|Map|Train explained fraction|Dev explained fraction|Dev cosine|
|---|---:|---:|---:|
|Aligned pre→post|.388244|.388764|.677852|
|Shuffled pre→post|−.003964|−.005921|.340445|
|Aligned post→pre|.400766|.395112|.759130|
|Shuffled post→pre|−.000936|−.000904|.543931|

Explained fraction is reduction in normalized-token squared error relative to
the aligned TRAIN target mean, NOT percent linguistic/semantic information.
The probe transfers, unlike a hypothetical train-only memorizer. It is still
linear, same-window and incomplete; negatives cannot establish non-invertibility.

Full519-gallery substitutions preserve ORIGINAL CLS/padding and frozen text;
therefore they are hybrid diagnostics, not encoder removal or compression models.
Identity channel delta1.5258789e-5≤2e-5 and all per-query ranks match exactly.

|Visual tokens|T2V R1|V2T R1|Mean R1|Mean delta pp|
|---|---:|---:|---:|---:|
|Identity|74.181118|76.300578|75.240848|0|
|Aligned forward prediction|51.637765|56.262042|53.949904|−21.290944|
|Shuffled forward prediction|6.551060|1.156069|3.853565|−71.387283|
|Train contextual mean|6.743738|.963391|3.853565|−71.387283|
|Post→pre→post roundtrip|65.510597|65.510597|65.510597|−9.730250|

Forward persistent mean rank worsens9.1413T/12.4598V; roundtrip2.2174T/4.0345V.
R5/R10 also worsen in both; full values are in the summary. Nonzero mean-control
R1 is NOT chance-calibrated: original informative CLS/padding and sequence masks
remain. Roundtrip starts from contextual outputs, so it cannot demonstrate raw
input sufficiency. This measurement supports recoverable shared structure, not
useful information discarded by the encoder. No novel candidate derived.

Completed exit0 in36.76s; peak allocated GPU386574848bytes. No crash/retry or
threshold change. Focused suite23 passed in1.04s. Previous goal turn classified
progress, current turn progress; objective remains active and no Proposal8.

### AS-C14 — separate video CLS, video pad and text pad paths

`AS-C14_protocol.md` precedes all8 fixed factor cells. Result/provenance
`AS-C14-MASK_run.json`; complete table and11/11 fallacy checks in
`AS-C14_result.md`. No fitting or selector. All seven nonidentity cells have
lower full-gallery mean R1; text-pad-only−1.445087pp, video-pad-only−.481696pp,
CLS-only−.385356pp. Joint exclusion reproduces AS-C01 at73.699422meanR1.
Nonadditive factor effects preclude attributing the joint drop to a single path.
Video-pad-only improves persistent mean ranks slightly despite worse overallR1;
this is disclosed, not hidden behind an all-metrics-negative claim.

Unchanged default rank parity passed; joint mask matches independent scorer
within3.8146973e-6. Runtime1.41s, GPU296833024bytes; exit0, no retries.
Focused suite24passed in1.04s; compileall/git diff --check clean. This supports
neither a mask-based method nor a claim that masked training would fail.
No GO/Proposal8; no global research-barrier claim. No live probe remains.

### AS-C15 — cohort dual support and numerical-tie attribution

Registered `AS-C15_protocol.md`, implementation `cohort_dual_audit.py`, three
tests. Run `AS-C15-COHORT-AUDIT_run.json` completed exit0 in29.98s. Detailed
interpretation `AS-C15_result.md`; all60 outcomes, dual certificates, hashes and
explicit post-hoc attribution in run/`AS-C15-SUMMARY.json`. No dev-label fitting.

Every assigned edge on all three519×519 matrices is a NON-STRICT row/column
maximum after subtracting label-free assignment-dual row/column potentials.
Reported primal-dual gaps, feasibility violations and assigned residuals all0.
This is not independent-rank reproduction: many ties remain; solver-specific
strict support is68–70% by row and1/519 by column. Potentials consume all queries.
Synthetic unchanged-focal-query/changed-other-query example proves cohort
assignment need not be invariant to other queries. No fixed-bank revival.

Twenty fixed±2e-5 matching perturbations per backbone retain R1 ranges:
seed42 82.080925–83.429672, seed1337 81.502890–82.851638,
seed2026 80.539499–81.695568. Mean gains7.379576/7.283237/6.127168pp;
average6.929994pp versus unperturbed7.803468pp. These are dependent numerical
stress realizations, not new trained seeds or a GO pilot. Original AS-C07
full-cohort R1 reproduced exactly before perturbing. All239/239/234 changed-row
occurrences swap identical deployed text inputs; no different-input swap.
This attribution is post-hoc and all outcomes remain disclosed. Official paired
positives unchanged; no linguistic-equivalence or PMGR permission inferred.

Bounded UPRet/SAN/CMCM code inspection is separately recorded in
`AS-C15_local_code_scope.md`, including existing modified UPRet versus committed
original distinction. No full-paper or whole-repository novelty audit claimed.
No new supported independent-query defect emerged from these checks.
Focused validation27passed in1.06s. Previous/current goal turns both progress;
objective active, no GO/Proposal8 or global exhaustion claim.

### AS-C16 — natural exact-lexicon control inventory

`AS-C16_protocol.md` preceded execution. Whole-word and deployed BPE multiset
keys independently find the same2 train groups/19 rows/61 pairs and no dev group.
All group IDs/texts and frozen paired margins in `AS-C16-LEXICAL_run.json`;
interpretation and11/11 checks in `AS-C16_result.md`.17 rows combine two evening
farewell wordings with “now” moved;2 rows move “usually” in a forecast. They
provide only TWO wording contrasts, not61 independent linguistic examples.
No signed-video or expert relational validity inferred. All61 pairs already
have strict margins in all four directions, minimum4.235100 across directions.
This demonstrates non-bag-of-words behavior on selected TRAIN instances, not
relational grounding, generalization or a new deficiency. No new probe trained.
Runtime1.86s, exit0, no retries. Two new tests passed.

### AS-C17 — same-weight nearby I3D views, limited diversity

`AS-C17_protocol.md`, `AS-C17-VIEWS_run.json`, `AS-C17_result.md` and
`AS-C17-SUMMARY.json`. Same R0, no training. Three canonical repeats match
cached tokens bitwise and all ranks; channel tolerance1.5258789e-5≤2e-5.
Nearby views42/1337/2026 mean R1=74.855491/74.759152/74.855491. Their fixed
mean74.759152 is−.481696pp versus canonical three-repeat mean75.240848.
It fails the predeclared exploratory lead gate; no method candidate generated.
Persistent mean ranks improve.271739T/.149425V; not an all-metrics failure.

379 videos change in each view,140 remain identical. Post-hoc exact-index audit
finds31 videos with3 distinct views,13 with2,475 with1; among changed videos335
have the same alternative across all seeds. The shared sampler's deterministic
fallback limits diversity; flag `view_independent` is relative to canonical only.
All475 identical-view score rows are identical across seeds. Negative averaging
does not establish temporal information sufficiency or global sampling failure.
No rescue sampler, seed sweep or closed sampling-family training launched.

Unchanged140 V2T score rows/ranks exactly preserved; eight corresponding T2V
query ranks change via other gallery videos, an explicit interference limit.
Runtime6.99s, GPU690465280bytes; run and summary exit0, no crash/retry.
All30 focused tests pass in1.11s, compileall/git diff --check pass. No live probe.
Previous/current turns progress; objective remains active, no GO or exhaustion.

### AS-C18 — shared video-parameter gradients including invalid slots

Protocol and code preceded run; six conditions completed in37.47s,
GPU20848029696bytes. Expired session was checked against final run JSON and
absence of a live process; not restarted, separate exit code unavailable.
`AS-C18-PARAMETER_run.json`, `AS-C18-SUMMARY.json`, `AS-C18_result.md` retain
all results, provenance, numerical limits and11/11 fallacy checks.

R0's86,296,064 declared visual parameters, inactive conv2_trans.weight explicit.
Exact encoder-cache parity and direct/chain8-example smoke (relative error0,
cosine1). Six reused AS-C12 first batches match losses/valid-interface cosines.
All6 parameter directional cosines positive (.116255–.203030). Duplicate-video
path signed projections.156002–.817325, invalid-slot.048181–.363366. Neither is
duplicate-query loss attribution, harmful update evidence or method validation.
Complement computed as total minus path; algebra check is not independent FP16
VJP closure. Fixed4096 loss scale, no actual update or dev/test access.

No new candidate, PMGR/RPCA remain closed.31 focused tests pass in1.20s.
Objective active, no Proposal8, no global exhaustion claim.

### AS-C19 — explicit generic initialization and internal source partition

Protocol `AS-C19_protocol.md`; audit `AS-C19-INITIALIZATION-V2_run.json`,
full limits `AS-C19_result.md`, indices `AS-C19-TRAIN-partition.json`.
Generic CLIP actual SHA40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
matches local lock. TRAIN extractor reports/checkpoint hashes verified for
BSL5K-agnostic/H2S-aware I3D; no individual feature rehash/pretraining overlap audit.
No PH/H2S/CSL retrieval checkpoint loaded; architecture config only reused.

300 generic tensors/148879617parameters copied exactly after dtype cast.
Explicit random projection/temporal positions/optional compression total844544;
two extra scalar-one parameters yield149724163 total. Repeated seed42 exact,
seed1337 changes every random tensor. Missing/unexpected keys fail closed.
Initial attempt failed at missing-key assertion because scalar declarations were
omitted. Failure/traceback retained; inspected literal-one declarations and
recorded V2 amendment; no tolerance change. V2 completed exit0 in10.79s,
GPU3434233344bytes. No optimizer update or model/cache artifact saved.

Fixed source-prefix hash fold0:5721fit/1375held,518/125prefixes,no overlap.
32text keys overlap,72held rows have fit-exact text. Source+text connectivity
inventory445components, largest1007/745; no alternative partition selected.
Prefixes inferred, not verified recording IDs; official splits/positives intact.
First32fit-row float32 smoke: clean loss3.629602,augmented3.644119,
gradient norm23.464052,149715969active gradient parameters; finite.
No heldout retrieval learning or useful residual supervision yet measured.

33 focused tests passed in1.21s; compileall/git diff --check pass. No live probe.
Next diagnostic requires learning calibration before correction/probe fitting;
new initialization is not automatically comparable to R0. AS-C07 distillation
rejection remains binding. Previous/current turns progress; no GO or exhaustion.
