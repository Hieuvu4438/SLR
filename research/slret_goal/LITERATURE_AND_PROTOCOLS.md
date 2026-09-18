# Literature and protocol map — partial, 2026-09-17

## Efficiency prior-art screen, 2026-09-18 (partial reading only)

[PLAID](https://arxiv.org/abs/2205.09707) uses centroid interaction/pruning for
ColBERT. [WARP v3](https://arxiv.org/html/2501.17788v3), abstract and selected
Section4 inspected, accelerates XTR with imputation/decompression/reduction;
Section4.3 explicitly does not guarantee its imputed bound. These are
AUTHOR_CLAIM/primary controlled-ML sources, not independent reproductions here.

[Lossless pruning v1](https://arxiv.org/html/2504.12778v1), abstract/introduction
and Section3.1–3.2 selected text inspected: exact dominance theory applies to a
modified ColBERT scorer (projection and ReLU); normalized original MaxSim cannot
generally drop a distinct document token for every possible query. Their practical
compression results and theoretical guarantee must not be conflated. This is not
an unchanged SEDS bidirectional softmax-mean theorem. SIGIR2025 metadata appears
in primary text; independent DOI/COI clearance not performed.

[EigenLI](https://arxiv.org/abs/2609.07561), abstract only, submitted7Sep2026,
uses document-specific spectral approximations for late interaction. Therefore
low-rank approximation itself cannot support a novelty claim. No absence-of-
prior-art claim; broader bound-search remains incomplete. Own decision: first
measure ordinary caching; do not admit a pruning method merely from these papers.

## Additional task/regime triage, 2026-09-18

[Scaling up Multimodal Pre-training, v1](https://arxiv.org/html/2408.08544v1):
selected method/data/implementation sections and Table X inspected, not a full
paper read. This does include sentence retrieval on PH and CSL. It uses
79-keypoint pose representation, multilingual MBart text, masked reconstruction
and contrastive pretraining on SL-1.5M. Table X provides the two added CSV rows.
This is a substantially different pretraining resource regime, not a matched
SEDS backbone comparison. Section III-A includes PH/CSL/H2S among corpus sources;
split exclusions and overlap manifests remain UNKNOWN, not proven leakage.
Implementation prose and Table III disagree on downstream learning rate;
do not resolve by guessing. AUTHOR_CLAIM, controlled-ML study/design-level III;
venue/version equivalence and COI not independently cleared.

[A Tale of Two Languages, v1](https://arxiv.org/html/2405.10266v1): selected
method and Appendix D text/tables inspected, not a full paper read. CSLR² also
performs sentence retrieval, but on BOBSL Sent-Val (2K) and Sent-Test (20K),
not PH/CSL/H2S. Frozen Video-Swin features and T5 text feed joint sentence/sign
retrieval training. Table A.8 supplies the separate Sent-Test CSV row; Table A.7
validation numbers must not be substituted. Appendix D.5 contrasts subtitle
alignment regimes, not a PH baseline intervention. AUTHOR_CLAIM,
controlled-ML study/design-level III; independent replication, venue/version
equivalence and COI not cleared. Own decision: retain as relevant prior art,
not a directly comparable leaderboard or permission to reopen lexical mining.

These are AI-assisted source-verification notes; neither source independently
confirms our local results. No novel-method or current-SOTA claim follows.

## SAN reading completion,2026-09-18

Primary[v1](https://arxiv.org/html/2607.09263v1): main text,Tables1–3,
references andAppendixA.1–A.4 read;figure captions read,visual panels not fully
inspected. AUTHOR_CLAIM: GFSLT-VLP teacher mines sign–word matches and visually
similar replacement words;100epochs SGD,CiCoB256/GFSLTB32. Fine-grained test
uses40 generated negatives per positive,not full-gallery retrieval. Appendix
adds POS filtering with GPT-4o-mini and SignDict examples scored by BSL1K-I3D;
these are not independent expert-validated semantic labels. Table1 CiCo coarse
T2V/V2TR1 changes69.2/70.1→68.1/67.8;GFSLT67.9/69.4→70.2/67.4. Therefore the
claimed favorable trade-off is not preservation within this campaign's0.5pp
directional guardrail. Own decision: no hard-negative mechanism reopened;
fine-grained gains cannot establish sentence-level SOTA. Publication now verified
in[ACL2026 proceedings](https://aclanthology.org/2026.acl-long.1302/),
pp28262–28277,DOI10.18653/v1/2026.acl-long.1302. Our table extraction remains
arXivv1;accepted-PDF version equivalence and COI not independently verified.

Current[CMCM publisher record](https://www.sciencedirect.com/science/article/abs/pii/S1077314225003546)
confirms CVIU264,104631,February2026,DOI10.1016/j.cviu.2025.104631 and official
code`https://github.com/vddong-zjut/CMCM`. Abstract/section previews cover
augmentation invariance,Gaussian alignment and temporal-motion covariance
pooling;full numerical tables still unavailable in retrieved preview. Do not
fill leaderboard from inference or reopen its numeric-repair lead.

No SOTA certification yet. Latest reading coverage2026-09-18: CiCo,SA,C²RL,
UPRet main text read;CiCo/SA included supplements and SAN main/appendix text
read. SEDS method/experiments read. Visual-panel coverage and separate
supplements remain incomplete where noted below;do not promote this to a
complete current-SOTA audit.

## SPOT-ALIGN reading,2026-09-18

[Primaryv2,main andAppendicesA–E](https://arxiv.org/pdf/2201.02495): full
extracted text read,including tables;visual panels and supplemental videos not
verified. AUTHOR_CLAIM: iterative dictionary/mouthing spotting trains I3D using
WLASL/MSASL exemplars and How2Sign. Retrieval freezes sign features,uses
average pooling,NetVLAD text and margin ranking;reported multi-seed variation
is retrieval-head training,not repeated backbone training. Best H2S combines
recognition-IoU with cross-modal scores;PH combination additionally uses an
SLT system. These are distinct supervision/inference regimes,not pure CiCo
baselines. H2S filtered counts31075/1739/2348 differ from later papers.
AppendixB:40epochs,RAdam,B128,selection by validation geometric meanR1/5/10.
AppendixD separates BOBSL recognition initialization from BSL1K dictionary
spotting initialization. Table5 changes both training and evaluation subtitle
alignment;its gap cannot isolate test-time corruption. Own decision: retain
single-model/combined rows separately;do not recycle lexical-support mining
or ensemble mechanisms. CVPR2022 venue verified in CVF metadata;source is
primary controlled ML evidence,not independent replication. Disclosed funding
includes Google/Adobe gifts;independent conflict assessment not performed.

## Published results, different resource conditions (AUTHOR_CLAIM)

Each tuple is R@1/5/10 in percent. These are published TEST results, never compared to our development or TRAIN diagnostics as a gain.

| Paper | Dataset | T2V | V2T | Source |
|---|---|---|---|---|
| SEDS | PH | 76.8/91.7/95.3 | 78.7/92.5/95.2 | [v1 Table2](https://arxiv.org/html/2407.16394v1#S4) |
| SEDS | CSL | 85.8/94.4/95.6 | 85.4/93.8/95.8 | [v1 Table3](https://arxiv.org/html/2407.16394v1#S4) |
| SEDS | H2S | 62.5/75.1/80.1 | 57.9/70.4/74.9 | [v1 Table1](https://arxiv.org/html/2407.16394v1#S4) |
| UPRet | PH | 72.0/89.1/94.1 | 72.0/89.4/93.3 | [v1 Table2](https://arxiv.org/html/2405.19689v1) |
| UPRet | CSL | 78.4/89.1/92.0 | 77.0/89.2/92.7 | [v1 Table3](https://arxiv.org/html/2405.19689v1) |
| UPRet | H2S | 59.1/71.5/75.7 | 53.4/65.4/70.0 | [v1 Table1](https://arxiv.org/html/2405.19689v1) |
| C²RL | PH | 78.7/92.2/94.9 | 77.6/91.3/94.2 | [v1 TableVI](https://arxiv.org/html/2408.09949v1) |
| C²RL | CSL | 90.3/96.4/97.7 | 88.4/95.7/97.1 | [v1 TableVI](https://arxiv.org/html/2408.09949v1) |
| C²RL | H2S | 62.4/75.9/80.1 | 57.5/68.4/73.0 | [v1 TableVI](https://arxiv.org/html/2408.09949v1) |
| SAN on CiCo, coarse | PH | 68.1/87.4/91.7 | 67.8/87.4/91.7 | [v1 Table1](https://arxiv.org/html/2607.09263v1) |

SEDS uses offline I3D, online SignBERT-initialized pose, CLIP text and dual contextual encoders. The paper describes common frame filtering/window sampling and24FPS; release preprocessing provenance still needs reconciliation. Current setup logs have lower test results than paper; they are historical observed evidence, not our new locked test.

C²RL includes context-generation representation training and MBart; treat it as a distinct pretraining/supervision regime until detailed comparability is established. SAN's fine-grained caption stress test is separate from full-gallery coarse retrieval; its headline fine-grained gain is not a full-gallery SOTA claim.

[CMCM publisher record](https://www.sciencedirect.com/science/article/pii/S1077314225003546) is a current candidate for the comparison map; complete methods/protocol/numbers remain UNKNOWN in this campaign. [Scaling up Multimodal Pre-training](https://arxiv.org/abs/2408.08544) and [A Tale of Two Languages](https://arxiv.org/abs/2405.10266) require task/regime triage; do not infer comparability from the word retrieval. SA source: [primary paper](https://arxiv.org/abs/2201.02495); CiCo: [primary paper](https://arxiv.org/abs/2303.12793).

## Locally matched protocol table

CiCo replay and SEDS adapted-input evaluation now have full official DEV-gallery evidence, with different feature regimes. PH DGS, English translated queries; 7,096 train,519 dev,642 official test count from metadata; canonical dev hash `3d44ee31477df729d7fb81d490cd8e7c4cd9f166fc3c5e8d3c7533fd3761ec0f`. Video/text IDs are paired one-to-one; identical strings do not create additional positives. Gallery519 each direction, independent pairwise scorer, no reranker/ensemble in these evaluations. CiCo padding/tokenizer/feature interpolation follow shared audited pipeline; max_words32, feature_len64, alpha.9. SEDS adapted recipe and provenance are separately locked in `ADAPTED_SEDS_PROTOCOL.md`; full TRAIN extraction is in progress.

Metrics preserve existing asymmetric ties: V2T double torch argsort; T2V exact-positive tie expansion. This differs from a generic optimistic best-positive evaluator; do not silently substitute. Medians for even populations require explicit torch versus NumPy handling. Real SEDS TRAIN256 has nine positive-tied V2T queries.

CiCo local strongest single model was selected on historical PH dev, uses BSL5K agnostic + H2S-transfer-aware features, and has mean R1 75.2408477842. Three-model uniform historical control77.263969 has not been reverified in this campaign. Neither number is a published TEST ranking. SEDS currently has complete TRAIN/test release assets but no official DEV release features; raw development data is available and requires a provenance-checked adaptation.

Still open: full dataset-specific gallery/dedup/pretraining tables; exact external checkpoint training-selection provenance; source/clip/pretraining leakage audit; CSL and H2S dev release-feature coverage; final literature version lock. Unknown is not contamination.
## Additional primary-source reading, 2026-09-17

CiCo v1 PDF: https://arxiv.org/pdf/2303.12793 (14 pages including supplement).
HTML endpoint unavailable; PDF sections3.2, parts3.3,4.1 and supplement A/B
partially read this turn, not yet a full-paper read. The paper specifies target
pseudo-label adaptation, unlike our PH cache's H2S-transfer-aware stream; retain
that protocol distinction. It describes BSL-1K pretraining while the public repo
names its downloadable asset `bsl5k.pth.tar`; do not infer contradictory corpus
provenance merely from the filename. Remaining method/experiment and supplement
coverage must still be completed before literature lock.

CiCo reading update: all main-text sections, references and supplementary A/B
text/tables now read from the primary v1 PDF, SHA256
`5a0bb54fe956baaad50d474c5a065d977801413f50c2b6dc5a2dfd9b3f44be44`.
Local page-anchor preflight is UNAVAILABLE (pypdf missing); cite section/table
identifiers, not certified local page numbers. Qualitative figure panels and
graph coordinates have not been fully visually verified. Tables1/2/3 supply
the added machine-readable CiCo rows. Supplement Table11 already tests outer
pooling; Table12 tests extraction stride. These are prior art, not new leads.
Translation, target-domain pseudo-labeling and supplementary ablations must
remain part of the comparability contract. No novelty inference from this read.

## Reading/audit continuation, 2026-09-18

SEDS v1 sections2–5, equations, tables1–6 and references read through the
[primary HTML](https://arxiv.org/html/2407.16394v1); introduction partly read,
figure images and any separate supplement remain unverified. Eq4 specifies
auxiliary weight.8, whereas current local `modules/modeling.py:forward` adds
both stream losses with weight1. Parser alpha is not consumed by that forward.
This is SOURCE_VERIFIED mismatch, not measured recall harm or admitted method.
Native continuation preserves weight1; no loss-weight sweep is authorized by
this observation. Table4 independent-stream ablations must not be equated with
our jointly trained branch readouts. H2S reported filtered counts31019/1738/2348
differ from raw catalog counts; exact manifest equivalence remains required.

UPRet v1 main text, equations, tables1–7 and references now read through the
[primary HTML](https://arxiv.org/html/2405.19689v1). Figure images and separate
supplement search remain pending. Sections3.6/4.3 describe distribution/OT as
training-only; inference runtime must therefore verify inactive branches rather
than assume an OT inference scorer. Domain-aware I3D uses target pseudo-labels;
H2S-transfer local features are a different regime. Table3 CSL T2V R10 and mean
rank do not both improve over the authors' CiCo reimplementation: report
trade-offs rather than general improvement in every metric. No reopening of
the historical transport-reduction mechanism.

These are AI-assisted primary-source reading notes, not independent reproduction
or full bibliographic/COI clearance. Both are controlled benchmark studies
(design-level III); paper result claims remain AUTHOR_CLAIM.

### VAP nearest-prior screen (2026-09-18)

[ECCV2024 primary](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05894.pdf):
intro/method and experiment text/tables1–7 inspected; references partly read,
figures/supplement unverified. VAP uses CoSign-1s skeletons, frozen mBART word
embeddings, greedy bidirectional token MaxSim contrast, and translation loss;
then pseudo-gloss-assisted translation adaptation and end-to-end fine-tuning.
Table6 reports PH TEST sentence retrieval for contextual-encoder and embedding
variants; added both rows, not translation BLEU. Directional trade-offs do not
establish superiority over CiCo. Gallery/dedup/tie/selection details remain
unverified; not a matched local baseline. Findings remain AUTHOR_CLAIM.

Decision (our inference): joint contrastive supervision of a visual backbone
and end-to-end adaptation are established prior art; RGB gradient caching is
only stronger-baseline feasibility. VAP is not evidence for SEDS recall gains.
Do not reopen pseudo-lexical supervision or generation/context stacking under
the VAP label. No candidate admitted from this reading.

### C²RL protocol completion (2026-09-18)

[Primary v1](https://arxiv.org/html/2408.09949v1): main text/equations/tablesI–IX
and references read; figure panels and any separate supplement remain pending.
[arXiv history](https://arxiv.org/abs/2408.09949) currently lists onlyv1;
no journal acceptance inferred from manuscript formatting.

ResNet18/ImageNet plus temporal convolution and three-layer Transformers learn
contrastive and autoregressive objectives jointly. Retrieval then uses offline
features and two independent MBart encoders, not a generation decoder at
inference. MBart vocabulary is trimmed on target TRAIN. Training: pretraining
200epochs/8×3090/batch64; downstream80epochs/batch128. Native-language MBart
queries differ from CLIP/translated-query baselines. Keep a distinct resource
regime, not a matched-backbone claim.

H2S filtered counts31085/1739/2348 differ from SEDS. PH TRAIN is written7098,
inconsistent with stated total8257; do not silently change canonical7096.
OpenASL TableVI says62.2 T2V R1, prose62.6: added table value with discrepancy
flag, not resolved by preference. TableVIII reports loss-weight ablations on
PH TEST; do not emulate this for our selection. Fig3 is described as robustness
across readout/backbone choices, not a held-out-domain robustness experiment.

Decision: C²RL informs comparability and prior art; generic generation/context
stacking remains inadmissible under existing registry. No candidate or GPU run
admitted. Controlled benchmark evidence/design-levelIII; AUTHOR_CLAIM only.

### Official supplementary-source coverage (2026-09-18)

SEDS introduction now read in the [arXiv v1 HTML](https://arxiv.org/html/2407.16394v1),
completing main-text reading, not figure or supplement inspection. The
[official OpenReview record](https://openreview.net/forum?id=RyPLu4tzU6)
confirms MM2024 Poster and exposes a supplementary ZIP link. Download failed;
the API returned HTTP403 ChallengeRequired. No challenge bypass attempted.
Thus the supplement exists but remains UNREAD, not absent. The HTML lists DOI
10.1145/3664647.3681237; publisher resolution returned403 in this session.
The offline-RGB memory limitation is already an author-stated motivation,
not a novel diagnosis from this campaign.

UPRet [official ECCV supplement](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06074-supp.pdf):
all extracted text, equations and references in sectionsA–C read. SectionA
derives entropic transport and Sinkhorn updates; this does not demonstrate
uncertainty calibration or linguistic correctness. SectionC reports TRAIN/DEV/TEST:
PH7096/519/642, CSL18401/1077/1176, H2S31164/1740/2356. Figure1 is a qualitative
How2Sign comparison; image panels have NOT been visually inspected (screenshot
attempt returned no visible image). Transport-reduction mechanisms remain closed.
The [Springer record](https://link.springer.com/chapter/10.1007/978-3-031-72784-9_22)
verifies DOI, ECCV2024/LNCS15102 and publication metadata; only the publisher
preview was accessed. The public ECVA main paper was partly read this session;
full equivalence to the previously read arXiv version is not established.

These are AI-assisted source-coverage updates, not independent reproduction,
full bibliographic clearance, human-read attestations or COI verification.
No experimental configuration or claim gate changes follow from this reading.

### Numerical-training prior-art screen (2026-09-18)

[Mixed Precision Training, v3](https://arxiv.org/abs/1710.03740v3): abstract and
version metadata read only; FP32 master weights and loss scaling already
address numerical limitations. [Adaptive optimizer quantization analysis,v2](https://arxiv.org/abs/2510.21314v2):
abstract/metadata read only; authors analyze optimizer-state quantization and
report sensitivity of Adam's second moments. Theorems/assumptions/experiments
not read; do not use it to prove underflow in our checkpoints. Both are primary
sources, not independent confirmation of this campaign. Thus FP32 optimizer
state is not an algorithmic novelty claim. A possible empirical study requires
measured causal retrieval effects, controls and replication beyond this screen.

2026-09-19 scoped follow-up supersedes the abstract-only reading status above:
see NUMERICAL_PRIOR_VERIFICATION.md for exact section coverage and caveats.
Mixed Precision Training main text is now read (figures not visually inspected).
The quantization paper's assumptions and theorem statements were inspected,
not its appendix proofs. Its applicability to our underflow mechanism was
rejected by the scoped source check; do not turn background into causal proof.
No novel-method admission or new GPU experiment results from this reading.
