# Mechanism collision audit — 2026-09-22

Bounded primary-source investigation of existing C27, plus a current-paper
scope check. This is not an exhaustive novelty review. External mechanisms are
[V] source-verified; external efficacy is [A] author-reported, never [R].

## C27 against external prior art

| Prior | Verified relevant content | Consequence for C27 |
|---|---|---|
| [CVT-SLR, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Zheng_CVT-SLR_Contrastive_Visual-Textual_Transformation_for_Sign_Language_Recognition_With_Variational_CVPR_2023_paper.pdf) | Sections 3.2.3–3.2.4 combine gloss CTC alignment and visual/text contrastive alignment for recognition. | CTC plus contrastive learning in sign video is established. Retrieval endpoint differs; that alone does not prove a new mechanism. |
| [CSLR² / A Tale of Two Languages, v1](https://arxiv.org/html/2405.10266v1) | Joint sentence/sign objectives; Table 3 tests auxiliary sign CE and SignRet with sentence retrieval. CE changes T2V/V2T R1 from 50.5/49.7 to 50.0/48.7; SignRet gives 51.7/50.2 under that BOBSL setting. Table 2 includes CTC without sentence retrieval. | Strong collision with a broad “sign supervision improves sentence retrieval” story; auxiliary design matters. Do not falsely state this paper already evaluates C27's exact CTC+SEDS configuration. |
| [SEDS, v1](https://arxiv.org/abs/2407.16394) and local `modules/modeling.py` | Current local forward has pose/RGB/fusion contrastive scores and optional cross-stream terms; no TRAIN gloss-label CTC. | C27 changes supervision, but is an extension of this retriever. An absence in SEDS code does not establish global novelty. |
| [SignSeek, September 2026](https://arxiv.org/html/2609.03695v1) | Gloss-based contrastive representation learning, articulator masking, dictionary retrieval and subtitle-alignment transfer. | Adjacent representation prior; task is isolated sign video-to-dictionary retrieval, not sentence T2V/V2T. Do not import its SOTA label into the project's frontier. |

Reading scope: CVT-SLR primary metadata and relevant method passages, CSLR²
joint-objective equations and Tables 2–3 with surrounding text, SignSeek abstract,
task definition and method overview. Not full-paper/figure/supplement reading.
The first CVT HTML and CSLR² v2 open attempts failed; primary PDF/index and v1
HTML supplied the cited passages. No inference of nonexistence from a failed URL.

Executed conceptual queries included `"sign language retrieval" "CTC"`,
`"sign language retrieval" gloss supervision contrastive`,
`"sign language retrieval" 2026`, and `"CVT-SLR" CTC contrastive variational
alignment`. Search discovery is not novelty proof. No private manuscript or
corpus was uploaded to another model.

## Eight-part semantic test of inherited C27

1. **Measured failure:** retrieval errors exist, but their attribution to missing
   ordered gloss evidence is [U]. TRAIN annotation availability is established.
2. **Input:** contextual pose windows plus independently annotated TRAIN gloss
   sequence, not a teacher-mined support mask.
3. **Output:** CTC auxiliary gradient; inference head removed.
4. **Changed parameters:** classifier plus whichever pose upstream parameters
   the future trainer explicitly allows. Actual trainer scope remains unspecified.
5. **Expected rank change:** better discrimination after unchanged SEDS fusion
   and scoring; unmeasured.
6. **Closest local pathways:** ELSC/SSSC auxiliary lexical supervision, closed
   temporal-order routes, and RPCA's training-only language auxiliary. The
   absence of support mining/protected updates is a difference, not a novelty
   certificate. Human ordered labels versus teacher support must be isolated.
7. **Closest external pathways:** CVT-SLR and CSLR² above.
8. **Separating experiment:** same labels/parameters/exposure, CTC versus
   order-free gloss content, both compared with native continuation. If only
   CTC versus no-gloss baseline is tested, the order-specific explanation is
   unidentifiable. An unexplained gain cannot pass a novelty claim.

Verdict: **OPEN**, high novelty risk, no primary-method selection. Cheap
diagnostics may continue; full pilot admission needs measured bottleneck,
adequate control and the current guide's remaining candidate-screen requirements.

## Cycle3 baseline collision boundaries

The [source/paper mechanism audit](evidence/BASELINE_MECHANISM_AUDIT.md) now
distinguishes four tempting routes before candidate generation:

- Learned token weighting plus a training-only Gaussian/transport auxiliary is
  already present in UPRet; inference omission is intentional. Local reduction
  repairs were already screened and do not supply a new mechanism.
- Visual-confusability caption substitution and a per-video hard-caption CE are
  already SAN's mechanism. Its constructed stress-task gain is not evidence of
  standard-gallery improvement or a reason to revive closed lexical mining.
- Contrastive content plus autoregressive context pretraining is already C²RL.
  A derivative translation kernel is not an official trained retrieval control.
- Augmentation adjustment, Gaussian alignment and temporal-motion covariance
  are CMCM collision targets. Incomplete public integration does not erase the
  prior or turn ordinary implementation repair into novelty.

These statements are grounded in the linked report's exact source functions
and primary passages. No global absence-of-prior claim or final-paper parity
is implied. The historical Q01/Q22 logical and positive-control checks also
remain binding in their actual scopes; do not relabel them as a new diagnostic.

## Cycle4 adjacent retrieval collisions

These targeted checks support the [six-route screen](09_CANDIDATE_SCREEN.md).
They establish precedents, not exhaustive novelty coverage or matched SLRet
efficacy. Primary CVF HTML and PDF direct opens returned403; author arXiv records
and primary-index excerpts supplied the bounded evidence below. No access failure
is used as evidence that a method does not exist.

| Source and reading scope | Verified mechanism | Screen consequence |
|---|---|---|
| [Thinking Fast and Slow, CVPR2021](https://arxiv.org/abs/2103.16553): abstract and primary CVF introduction excerpt | Pair cross-attention retrieval combined with fast dual-encoder distillation/reranking | P2's generic pair-verification route is established. No exact equivalence of all possible sign-specific operators is asserted. |
| [TokenBinder, WACV2025](https://arxiv.org/abs/2409.19865): abstract and primary CVF introduction excerpt | One-to-many candidate comparison through focused-view cross-attention in a coarse/fine framework | P6 cannot claim candidate-set comparison as novel. This prior was also present in the dataset-first literature map; the current pass corroborates it. |
| [m-RNN, 2014, §6](https://arxiv.org/html/1410.1090v1): abstract, training equations5–6 and retrieval section | Conditional sentence likelihood is used for retrieval; sentence ranking divides by a sentence prior estimated using TRAIN images | P4's generic conditional-likelihood/prior-normalization score is directly anticipated. Image retrieval is adjacent, not evidence of sign-language efficacy. |

Analytic consequence for P4, not an empirical result: for fixed query text t,
subtracting log p(t) from every video's log p(t|v) leaves T2V ordering unchanged.
For fixed video v it can alter V2T ordering. This does not claim that the full
conditional decoder score equals the incumbent similarity score. It rules out
the narrower claim that text-prior normalization itself repairs both directions.
Length normalization and vocabulary change the scorer and require disclosure;
neither automatically supplies novelty.

Queries executed included cross-encoder/fine-grained video-text retrieval,
generative-likelihood sentence/sign retrieval, simultaneous-articulator alignment,
and exact title searches with cross-attention/one-to-many terms. Other discovered
papers, including EagleNet, are leads only and are not cited as verified mechanisms
here. P3's exact relation-specific external novelty remains OPEN; its generic
form already fails local-separation and measured-bottleneck gates. P5 inherits
the proposal7 primary-source mmSampler collision, not a fresh full-paper review.

## Cycle6 external diagnostic precedent

The [ASL-MTP source/scope record](evidence/ASL_MTP_DIAGNOSTIC_SCOPE.md) adds a
bounded precedent for linguistic contrast evaluation with cue ablations, including
matched training/inference conditions. It does not provide PHOENIX relevance
labels or support a new relational head. Existing P3 and Q17 admission decisions
remain unchanged. This is a diagnostic/prior-art lead, not a sentence-retrieval
frontier result or an executable local benchmark.

## Cycle11 handshape supervision boundary

[PHOENIX14T-HS source check](evidence/HANDSHAPE_RESOURCE_SCOPE.md) adds a
concrete collision target for generic handshape auxiliaries and an annotation
validity warning for Q04/Q17. It does not establish an equivalent retrieval
mechanism, a PH video-relevance oracle or a stronger fair retrieval comparator.
No candidate is promoted by the presence of a published label resource.

## Cycle13 pooling-prior transfer boundary

[Source-to-implementation screen](evidence/POOLING_PRIOR_TRANSFER_SCREEN.md):
the inspected SEDS scorer already uses soft expectation, and the local record
already covers the obvious pooling and channel-objective changes. The newly
checked adjacent source does not establish a new local bottleneck or support
reopening those interventions. Its displayed concentration equation also needs
qualification; no author-code defect or invalid empirical result is inferred.
This is source qualification, not progress toward mechanism admission.

## Cycle14 paired-anchor collision

[Paired-anchor screen](evidence/PAIRED_ANCHOR_SCREEN.md) checks a distinct
fixed-TRAIN-bank construction before any extraction. It is not universally a
candidate-only offset, but its generic information path is already explicit in
RELIT (ACL2023). No observed SLRet-specific bottleneck or separating intervention
was supplied. Reject this generic proposal; do not miscite the older forced-bank
assignment or gallery-filtering negatives as direct tests of this formula.
