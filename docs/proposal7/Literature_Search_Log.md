# Literature investigation log

## 2026-09-16: first user-authorized selective reopening pilot

User explicitly approved selective reopening with justification and controls.
Revisited [X-Pool primary methods](https://arxiv.org/html/2203.15086v1) as adjacent
prior for learned sentence conditioning, not a novelty clearance or SOTA claim.
[CICO-REOPEN-01](evidence/autonomous_search/CICO-REOPEN-01_result.md) implemented
and trained a narrow outer-weighting probe with blind/random-conditioning controls.
No helpful lead: all select unchanged CiCo initialization. No new broad search,
author-result reproduction or TEST-driven decision. This records experiment
consequences of the existing prior, not a new literature-discovery cycle.

## 2026-09-16: CiCo-first development, query-conditioning transfer boundary

User now prioritizes CiCo implementation and comparison with SEDS/other SLRet.
[Focused build plan](CICO_FOCUSED_BUILD_PLAN.md) records the source relationship
and the next implementation gates, not another general survey.

New targeted primary reading: [X-Pool](https://arxiv.org/html/2203.15086v1),
§§4.2–4.4. Learned query-conditioned pooling exists in adjacent retrieval work.
CiCo's own inner soft expectation already equals a query-token-weighted visual
vector dotted with that token. Thus generic query conditioning is not itself a
missing capability. This does not equate full X-Pool and CiCo function classes.
Official CVF search record verified identity; page open403, arXiv methods available.
SEDS architecture context refreshed from [its primary paper](https://arxiv.org/html/2407.16394v1)
and local modeling/fusion source. No reported recall selected an experiment.

Search strings: CVF X-Pool video text retrieval; SEDS semantic enhanced sign;
CVF2025 fine-grained video-text alignment. Other hits were discovery-only,
not read papers, verified technical claims or shortlisted methods. No new
systematic review, novelty clearance, runtime replication, assets or training.

## 2026-09-16: repository overview and UPRet objective-capacity bound

Compiled [repository-gap overview](evidence/autonomous_search/REPOSITORY_GAP_OVERVIEW.md)
from the existing bounded source/measurement reports, separating defects from
unresolved scientific gaps and paper-adaptation prerequisites. Not a new broad
literature survey, exhaustive code audit, current-SOTA table or method GO.
The official [SLRT README](https://github.com/FangyunWei/SLRT#readme) confirms
CiCo belongs to that umbrella; other task implementations are not independent
sentence-retrieval baselines. Local README and relevant pinned SAN source paths
were inspected; no TEST loader was executed.

Rechecked [UPRet arXivv1](https://arxiv.org/html/2405.19689v1), §3.6 and inference
efficiency paragraph: OT is intentionally training-only. New source-derived
auxiliary-score bound and posthoc inference from saved TRAIN no-OT losses are in
[the result](evidence/autonomous_search/UPRET_loss_rank_bound_result.md).
Primary methods support design intent, not the new algebra or measured efficacy.
No final-version re-review/full-paper claim. Published performance tables appeared
in browser context but were not used for hypothesis/candidate selection.
No new assets, training, SEDS use, TEST content or external model calls.

## 2026-09-16: official alignment provenance gate

Official RWTH2016directory lists PH2014T alignment archives. Read PAMI2019author
methods (§3opening,§§4–4.2) and Re-Sign README/three source files at
b1c7981ed51aace2c8c84a9ddf30d06500509f1c. Weak-label iterative timing, not an
independent human cue certificate. Exact archive contents/join unverified; none
downloaded. Searches used both exact alignment archive names. No benchmark
results used; browser context included published tables incidentally. Full
[provenance and limits](evidence/autonomous_search/PH_alignment_provenance_gate.md).
No dataset/examples/models/TEST assets or training; no method promotion.

## 2026-09-16: complex TRAIN annotation semantics

Read official2012annotation conventions completely (2pages) and local release
README/evaluation shell. Queries: RWTH PHOENIX filename/original-recording mapping;
exact complex CSV filename; `RWTH PHOENIX complex annotation mouth train`.
Official codebook located and retrieved; original detailed fields must not be
assumed present in transformed release. Literal TRAIN census retains coarse
markers and neg- but finds no original detailedpostfixfields. No cue-isolating
labels or visualinformationlossclaim follows. [Full scope/counts](evidence/autonomous_search/PH_complex_notation_result.md).
No TEST contents, other archive acquisition, model run or method promotion.

## 2026-09-16: CMCM Gaussian alignment contract

Local pinned CCG_Module.py inspected completely; unchanged source yields negative,
unbounded matched-distribution loss due to asymmetric epsilon placement. Six
CPUfixtures plus exact derivative verified. This is not a novel Gaussian model or
a trained retrieval effect. GitHub HTML/raw opens failed; local pin/hashes retained.
PyTorch2.11 distributions API retrieved for reference implementation identity.
Search `"Sign-Aware" "SAN" "retrieval" github` used for discovery; no paper results
used for this diagnostic. See [result](evidence/autonomous_search/CMCM_gaussian_result.md).
No TEST/DEV data, trained assets, variance census or training campaign.

## 2026-09-16: raw-input padding source exposure check

Pinned local SLRT38a4f7b source plus deterministic local extractor inspected;
browser open of pinned GitHub videodataset.py failed (cache miss), no successful
live retrieval claimed. No new paper or novelty claim. TRAIN-only metadata census
finds0/7,096videos below16frames and0window-geometry mismatches across720,914windows.
Thus short-input repeat-last padding is inactive in current TRAIN extraction.
Internal convolution padding/decoder failures/pretraining remain outside scope.
See [result and read limits](evidence/autonomous_search/CICO_input_padding_result.md).
No TEST contents/model assets, method promotion or padding-repair sweep.

Status: **TWO-CYCLE SEARCH RECORDED; bounded coverage, not exhaustive**. Search began only after the historical-document audit,
negative-results registry, and relevant implementation-path audit. No candidate
method has been selected. This is a targeted, primary-source research audit, not
an exhaustive PRISMA systematic review.

## Scope and actual cutoff

Search/check date: **2026-09-14**, Asia/Ho_Chi_Minh. Include sentence-level
text↔sign-video retrieval, closely related representation learning, and prior
mechanisms relevant to novelty collisions. Exclude recognition/translation-only,
isolated-sign dictionary retrieval, word-presence detection, and sign-video↔video
corpus search from the main sentence-retrieval leaderboard. These may remain
mechanism references. Separate published author reports [A], verified public
source behavior [V], new local measurements [M], inference [I], hypothesis [H],
and unresolved information [U].

## Executed queries

All below were actually issued on 2026-09-14. Quotes are query syntax, not paper
quotations. Search results are discovery leads; technical claims require reading
the original paper or official implementation.

1. `"sign language retrieval" 2026`
2. `"sign language video retrieval" 2025 2026`
3. `"sign-aware" "negative" retrieval SAN`
4. `"Causal" "Multi-grained" "sign language" retrieval CMCM`
5. `"sentence-level sign language retrieval" 2026`
6. `"text to sign video retrieval" 2026`
7. `"sign language video text retrieval" 2025 2026`
8. `"sign language representation" "retrieval" 2026`
9. `"C2RL" "github"`
10. `"Causality-inspired multi-grained cross-modal sign language retrieval" results`
11. `"fine-grained" "sign language retrieval" 2026`
12. `"cross-modal sign language retrieval" 2026`
13. `"Content and Context Representation Learning" github Chen`
14. `"Content and Context Representation Learning" "2025" IEEE`
15. `"sign language retrieval" "2026" -site:researchgate.net -site:eurekamag.com -site:themoonlight.io`
16. `"sign language retrieval" "2025" -site:researchgate.net -site:eurekamag.com -site:themoonlight.io`
17. `"Content and Context Representation Learning" "github" "Zhigang"`
18. `"Content and Context Representation Learning" "10.1109"`
19. `site:ieeexplore.ieee.org "3553052"`
20. `"SPOT-ALIGN" "github.com"`
21. `"CiCo" "sign language retrieval" "2026" -"Semantic Hardness"`
22. `"C2RL" "retrieval" 2026`
23. `"sign language" retrieval "Dual" "Anchor"`
24. `"2607.27614"`
25. `"2608.20473"`
26. `"sign language retrieval" "2026" "alignment" -"Semantic Hardness" -"Causality-inspired"`
27. `multilingual video text retrieval native language translation multilingual CLIP MMP retrieval`
28. `sign language recognition 3D hand reconstruction inverse graphics MANO`
29. `adaptive frame sampling video text retrieval active acquisition`
30. `generative video text retrieval conditional likelihood diffusion density ratio`
31. `likelihood ratios out of distribution detection deep generative models Ren 2019 background`
32. `contrastive predictive coding density ratio InfoNCE p x context p x`
33. `"sign language retrieval" "likelihood"`
34. `"sign language retrieval" "information" "2026"`
35. `"Sign Language Video Retrieval with Free-Form Textual Queries" "2025"`
36. `"Uncertainty-aware Sign Language Video Retrieval" "2026"`
37. `"Semantically Enhanced Dual-Stream Encoder" "2026"`
38. `"Content and Context Representation Learning" "retrieval" "2026"`
39. `"Graph traverse reference network" sign language arxiv`

The 2026 searches found SAN and CMCM as directly relevant new papers. This does
**not** establish that no other relevant paper exists. Some broad queries returned
irrelevant finance or signed-network records; these were excluded. ResearchGate,
Moonlight, blogs, catalogs, and bibliographic aggregators were used only as leads,
not as technical evidence or independent replications.

## Verified source records and current reading coverage

### SPOT-ALIGN

Duarte, A., Albanie, S., Giró-i-Nieto, X., & Varol, G. (2022).
*Sign Language Video Retrieval with Free-Form Textual Queries*. CVPR.
[Official project](https://imatge-upc.github.io/sl_retrieval/).

[V] Project read completely; [arXiv v2](https://arxiv.org/pdf/2201.02495)
main method/settings/results and most appendix read. Remaining gaps are minor
related-work/reference/figure-label passages; no claim of full visual inspection.
[A] v2 H2 combined R1 is **32.8/23.3**, not later tables' 34.2/23.6;
the version difference is unresolved. Retained train count 31,075 differs from
CiCo's 31,085. [U] Official training repository not found. Project discloses public
research funding and Google/Adobe gifts; these are disclosures, not evidence of
bias. Do not reuse inconsistent page ranges from later papers.

### CiCo

Cheng, Y., Wei, F., Bao, J., Chen, D., & Zhang, W. (2023).
*CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning*.
CVPR. [Author-hosted PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/06/CiCo.pdf),
[official repository](https://github.com/FangyunWei/SLRT).

[V] PDF retrieved: 14 pages; main paper through results and appendix method/
ablations read. References and remaining figure labels not fully read. This
author-hosted arXiv-v1 document is not certified identical to final CVPR text.
Existing local official-code audit
is separate from that reading coverage. Domain adaptation uses pseudo-labels;
contrastive matching is soft token correspondence, **not** a hard one-to-one
assignment. The latter is corroborated by the audited implementation, not inferred
from later papers' characterizations.

### UPRet

Wu, X., Li, H., Luo, Y., Cheng, X., Zhuang, X., Cao, M., & Fu, K. (2024).
*Uncertainty-aware Sign Language Video Retrieval with Probability Distribution
Modeling*. ECCV. [Paper v1](https://arxiv.org/html/2405.19689v1),
[official repository](https://github.com/xua222/UPRet).

[A] Gaussian feature modeling plus OT; Eq. 24 explicitly uses the OT score only
during training. R@1 T2V/V2T: PH 72.0/72.0; H2 59.1/53.4; CSL 78.4/77.0.
Paper states 200 epochs, batch 512, four A100s, CLIP initialization and dual I3D
features. [V] Main method, experimental settings, main tables and ablations read;
references partially read. [U] Final ECCV-versus-arXiv textual parity and exact
author-run selection provenance remain unresolved. The public implementation has
already required local compatibility/RNG repairs; those do not establish a
successful independent reproduction.

### SEDS — literature/source only; pretrained assets prohibited

Jiang, L., Wang, M., Li, Z., Fang, Y., Zhou, W., & Li, H. (2024).
*SEDS: Semantically Enhanced Dual-Stream Encoder for Sign Language Retrieval*.
ACM MM. DOI: 10.1145/3664647.3681237.
[Paper](https://arxiv.org/html/2407.16394v1),
[official source](https://github.com/longtaojiang/SEDS).

[A] Online pose GCN plus frozen RGB I3D, local cross-modal attention and
pose–RGB matching. SignBERT hand initialization, RTMPose, and CLIP are additional
resource/provenance dimensions. R@1 T2V/V2T: PH 76.8/78.7; H2 62.5/57.9;
CSL 85.8/85.4. H2 retained counts are **31,019/1,738/2,348**, differing from
CiCo/C²RL train/dev counts despite equal nominal test size. [V] All main text,
settings/results/ablations read; references partly pending. [U] Equal test count is
not verified equal ordered gallery. No SEDS checkpoint, teacher or feature asset
was downloaded or loaded. Industry affiliation is disclosed; a COI declaration
has not yet been located.

### C²RL

Chen, Z., Zhou, B., Huang, Y., Wan, J., Hu, Y., Shi, H., Liang, Y., Lei, Z., &
Zhang, D. *C²RL: Content and Context Representation Learning for Gloss-free Sign
Language Translation and Retrieval*.
[Author preprint, 2024-08-19](https://arxiv.org/html/2408.09949v1).

[A] Contrastive content plus autoregressive context pretraining; ResNet18/
temporal convolution and separate mBART retrieval encoders. Table VI R@1 T2V/V2T:
PH 78.7/77.6; H2 62.4/57.5; CSL 90.3/88.4. H2 filtered counts
31,085/1,739/2,348. Pretraining uses 200 epochs on eight 3090s; retrieval
fine-tuning 80 epochs. PH ablations are explicitly test-set results; selection
provenance needs checking. [V] Main text/settings/tables read; references partly
pending. [U] Final TCSVT 2025 version needs primary-publisher verification;
DOI 10.1109/TCSVT.2025.3553052 discovered in DBLP, but publisher resolution failed.
No official code located in executed searches. A third-party SLT implementation
is not an official SLRet reproduction. COI information not located.

### SAN

Lee, J., Hur, C., Choi, C., Cho, S., Gaim, F., Hwang, E. J., Song, H., & Lim,
K. (2026). *Semantic Hardness Is Not Visual Hardness: Sign-Aware Hard Negative
Mining for Sign Language Retrieval*. ACL, 28262–28277.
[Official record](https://aclanthology.org/2026.acl-long.1302/),
[paper v1](https://arxiv.org/html/2607.09263v1),
[official source](https://github.com/joonmy/SAN).

[A] Mines visually close, differently labeled sign–word pairs and substitutes
caption words. PH-only stress gallery: original caption plus 40 generated
negatives. CiCo stress R@1 rises 17.9→39.4, while standard T2V/V2T falls
69.2/70.1→68.1/67.8. GFSLT stress rises 16.8→49.1; standard
67.9/69.4→70.2/67.4. These support a stress-test gain, not full-gallery SOTA.
Confusability uses learned-embedding proxies, not sign-level ground truth;
test-target construction does not alone prove training leakage. [V] Main text,
limitations/ethics/funding and appendix read; a short reference-list gap remains.
Government research funding
is disclosed; independent cross-language validation is absent. Public-code issues
below must not be silently generalized to the authors' actual runs.

### CMCM

Yang, X.-H., Wei, D., Li, W., & Hu, H. (2026).
*Causality-inspired multi-grained cross-modal sign language retrieval*.
Computer Vision and Image Understanding, 264, 104631.
[Publisher](https://www.sciencedirect.com/science/article/pii/S1077314225003546),
[official source](https://github.com/vddong-zjut/CMCM).

[V] Publication is February **2026**, although DOI
10.1016/j.cviu.2025.104631 contains 2025. Publisher preview read. [A] Augmentation
backdoor adjustment, Gaussian causal attention, and temporal-motion covariance
pooling; three-dataset comparisons. [U] Full numeric tables and complete
identification assumptions not accessible; no numerical leaderboard entry is
invented. A neural module named “causal” does not establish causal identification.
Authors declare no competing interests and disclose Chinese public funding.

## Official repository checks, pinned 2026-09-14

New clones are confined to `third_party/`; no upstream source was edited.

| Repository | Commit | Verified availability boundary |
|---|---|---|
| SAN | `82aba9cbc1beb403abef6e9a3875ca52479805c8` | README says Coming Soon; training/model/data code present, negative table is an unspecified path; no mining-construction script in tracked tree |
| CMCM | `5d458719d1da2f082e188cc44705003d919e7e97` | Modules/datasets present; no tracked training driver, experiment configuration, result table or README |

Reproducible evidence: `tools/audit_new_upstreams.py` →
`evidence/new_upstream_audit.json`, recording source hashes, tracked-file inventory,
and synthetic CPU probes. It does not run training, download weights, or unpickle
dataset/model assets.

SAN [V]: `train_vlp_v2.py:191` builds the loader from `test_label_path`;
line 273 evaluates that loader for epoch selection despite `dev_*` variable and
checkpoint names. The supplied YAML points to `labels.test.cleaned`; its dev
path is not consumed by this trainer. This default path violates a dev-only
selection contract. The released implementation uses German BERT text and an
mBART-form visual encoder, so filename/class names do not establish CiCo parity.
Its scorer also leaves inner attention padding unmasked. Published outcomes
remain [A]; actual author execution/configuration is [U].

CMCM [M]: constructing `CausalBackdoorAdjuster` fails with undefined `DEVICE`.
`CausalAttention(8,2)` fails on `[2,64,8]` because its 1024² mask slices the
wrong axes; the `[1,1024,8]` control returns successfully. [V] `Encoder.py`
names an R(2+1)D classifier `i3d_encoder`, leaves the supplied checkpoint argument
unused, and has unresolved sequence/projection dimensions. These public files
are not an executable paper-reproduction package. They are not evidence that
the reported method itself was never implemented elsewhere.

## Citation expansion and screened adjacent leads

Backward links read in SAN include CiCo, UPRet, SEDS, SPOT-ALIGN, *Verbs in
Action*, *Beyond Coarse-grained Matching in Video-Text Retrieval*, TripletCLIP,
and phonological minimal-pair work. CMCM explicitly references C²RL. Fresh
forward-keyword searches for CiCo/C²RL surfaced *Selective Contrastive Learning
For Gloss Free Sign Language Translation* (ACL 2026), *SignGPT and the Visual
Language Toolkit*, and *Gloss-Free Sign Language Translation: An Unbiased
Evaluation of Progress in the Field*. Original-source mechanism reading remains
pending; these are **leads, not adopted evidence**.

Other screened leads requiring task check: *Graph traverse reference network for
sign language corpus retrieval in the wild* (Neurocomputing 2025), *SignRep:
Enhancing Self-Supervised Sign Representations*, and *CCC: cross-modal contrastive
creator for end-to-end sign language generation* (published December 2025).
WSLP 2025 *Pose-Based Sign Language Spotting via an End-to-End Encoder
Architecture* explicitly evaluates binary word presence, not the target
sentence-retrieval gallery; exclude from the main leaderboard.

## Source grading and limits

This is ML research, so grades concern controlled benchmarks, reproducibility,
resource fairness and external validity—not a biomedical RCT hierarchy. Venue
peer review is not independent replication. Provisional grading: CiCo/UPRet/SEDS
are strong primary method sources but only moderate evidence for a transferable
effect until common-protocol reproduction; SAN is moderate evidence for its
specified PH stress task and weak evidence for standard-retrieval improvement;
C²RL is a primary preprint with strong reported results but unresolved final-code
provenance; CMCM has verified publication but insufficient accessible evidence
for numerical ranking. No unverified-reference title is promoted into the final
bibliography. No citation-count, retraction-database, or comprehensive forward-
citation audit has yet been completed or claimed.

AI-assisted search, source inspection and code diagnostics were used. No human
sign-language expert validation or independent reviewer panel has occurred.
The four-candidate collision search and inline four-perspective review are now
recorded in `Research_Questions_and_Candidate_Screen.md`. A second cycle added
input-channel/seed-persistence measurements and density-ratio reduction tests.

## Additional verified primary sources and exclusions

- Zhou, W., Zhao, W., Hu, H., Li, Z., & Li, H. (2024 preprint),
  [Scaling up Multimodal Pre-training for Sign Language Understanding](https://arxiv.org/html/2408.08544v1):
  §III-C, §IV-A, Table X and scale ablations read. [A] 79-joint pose encoder,
  SL-1.5M pretraining (100 epochs, eight A100s); R1 PH 74.5/75.1,
  CSL 87.5/87.2. Track C, not equal-resource. Prose/table disagree on
  optimizer/weight decay and fine-tuning LR; exact public implementation not
  checked. H2 table is translation, not SLRet. Full paper not claimed read.
- [DualAnchor](https://arxiv.org/abs/2607.27614), 2026-07-30: abstract/metadata
  read; LLM prior and partial OT for SLT, not main retrieval leaderboard.
- [AVIOT](https://arxiv.org/abs/2608.20473), v1 2026-08-20/v2 2026-08-25:
  abstract/metadata read; video-LLM token compression, not sentence SLRet.
- [Selective Contrastive Learning For Gloss Free Sign Language Translation](https://aclanthology.org/2026.acl-long.2116/),
  ACL 2026: abstract/metadata read; negative-selection trajectories/curriculum,
  not sentence SLRet. Does not reopen closed negative-learning families.
- [Gloss-Free Sign Language Translation: An Unbiased Evaluation of Progress in the Field](https://arxiv.org/abs/2603.13240):
  abstract/metadata read; evaluation caution concerns SLT, not a direct
  refutation of any SLRet result. Repository not newly audited.
- Asasi, S., Mercanoglu Sincan, O., & Bowden, R. (2026-09-03),
  [SignSeek](https://arxiv.org/abs/2609.03695): abstract/metadata read;
  dictionary video-query retrieval, articulator masking and same-gloss
  pretraining. Not sentence text↔video retrieval. No assets loaded.
- Lei, J., Berg, T. L., & Bansal, M. (2021),
  [mTVR](https://arxiv.org/abs/2108.00061): abstract and official repository
  README read; multilingual parameter sharing is prior art for candidate A.
- Hu, H., Zhou, W., & Li, H. (2021),
  [Hand-Model-Aware Sign Language Recognition](https://cdn.aaai.org/ojs/16247/16247-13-19741-1-2-20210518.pdf):
  abstract through Our Approach/objectives read; differentiable MANO,
  geometric consistency and recognition precede candidate B. Not SLRet.
- Hu, Z., Ye, N., & Mohomed, I. (2022),
  [mmSampler](https://proceedings.mlsys.org/paper_files/paper/2022/hash/d59a1dc497cf2773637256f50f492723-Abstract.html):
  official abstract read; learned compute-saving frame policy precedes C.
- Jin, P., et al. (2023), [DiffusionRet](https://arxiv.org/abs/2303.09867):
  abstract and official ICCV introduction read; generative video retrieval
  precedes D, not proof of identical conditional-density equations.
- Ren, J., et al. (2019),
  [Likelihood Ratios for Out-of-Distribution Detection](https://papers.neurips.cc/paper_files/paper/2019/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html):
  official abstract and retrieved §3 excerpt read. Background likelihood
  correction is not a new general principle; OOD detection is not SLRet.
- van den Oord, A., Li, Y., & Vinyals, O. (2018; v2 2019),
  [CPC](https://arxiv.org/html/1807.03748v2): §§2.1–2.3 read. Density-ratio
  interpretation of contrastive scoring is prior art, not a guarantee that
  finite-capacity CiCo estimates a calibrated true density.

Final task checks: [GTRN author-institution record](https://ro.ecu.edu.au/ecuworks2022-2026/6028/)
and publisher search content identify visual-sign-query to video-document retrieval,
not sentence text↔video (direct publisher open first returned 403).
[CCC](https://link.springer.com/article/10.1007/s44443-025-00418-3), abstract and
§4.4, is generation; MAX/MIN R1 summarize a training trajectory, not a locked
dev-selected standard-gallery result. [SignGPT/VLTK](https://discovery.ucl.ac.uk/id/eprint/10225233/1/26026.pdf),
abstract/overview, is a corpus/recognition toolkit; no sentence retrieval score
was identified in the inspected material. Exclude these from the main table.

Unresolved: inaccessible CMCM tables, final-version parity for some preprints,
official C²RL code, full citation-graph coverage,
and retraction-database verification. These prevent a claim of **exhaustive
current SOTA certification**; they do not justify fabricated numbers or novelty.
Search stop is a bounded two-cycle proposal decision, not proof that no better
method or unlocated paper exists. Browser PDF screenshots failed for SPOT/CiCo;
no claim of visual verification. No local new PDF page locator is trusted without
the required preflight; final citations use visible section/table locators.

## 2026-09-16 incremental transfer-scope screen

New primary source: Wong, R., Jang, Y., Momeni, L., Varol, G., & Zisserman, A.,
[*SignMatch: Matching Dictionary Signs to Continuous Sign Language Video*](https://arxiv.org/html/2609.01886v1).
Metadata corroborated by the Oxford publication record. Read scope, exact task
boundary, resource differences, transfer decisions, search queries and limitations
are in [the scope screen](evidence/autonomous_search/SignMatch_transfer_scope_screen.md).
Adjacent prior only; no sentence-retrieval performance or novelty inference.
No implementation audited, new assets acquired or candidate experiment launched.
This incremental search is not exhaustive current-SOTA certification.

## 2026-09-16 human TRAIN consistency resource

Czehmann et al., *“A Sacred Bird Called the Phoenix”* (LREC2026 sign-language
workshop), [primary record](https://www.sign-lang.uni-hamburg.de/lrec/pub/26064.html)
and [author repository](https://github.com/DFKI-SignLanguage/sacre-bird-phoenix).
New pinned TRAIN annotation join and §4.1 sampling audit are in
[the resource result](evidence/autonomous_search/SACREBIRD_train_join_result.md).
307/307exactTRAIN identifiers; no DEV match. Gloss/German judgments, not video
relevance, English-caption validity or error-free controls. Structured sample,
one annotator. Only TRAIN CSV fetched, TEST CSVs unopened. No full-paper claim.
Queries: PHOENIX2014T negation/nonmanual annotations; semantic numbers/spatial
annotations; linguistic/HamNoSys annotations; exact paper title with training;
Czehmann307Phoenix. Secondary sources used only for discovery, not technical
evidence. Primary HTTP fallback and PDF-extraction failure disclosed in result.

Follow-up [semantic-scope check](evidence/autonomous_search/SACREBIRD_semantic_scope_result.md):
all307TRAIN comments read through150distinct strings; primary §3 methods/codebook
read, not full paper. Reordering labels explicitly excluded. First fixed original
TRAIN triple corroborates gloss/German disagreement while English matches German;
signed-video truth remains unidentified. No new labels, model diagnostic, TEST
CSV access or noisy-label method. This completes the two-pass resource lead.

## 2026-09-16 masked-query interpretation check

[Set Transformer](https://arxiv.org/html/1810.00825v3), §3.2 and abstract/metadata:
learned-seed attention pooling precedes a generic fixed-query-bank replacement.
[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588), abstract
only: adjacent artifact motivation, not evidence of CiCo padding contamination.
Source-derived differences and limits are in the
[visual-mask result](evidence/autonomous_search/CICO_visual_mask_result.md).
No full-paper reading or exhaustive prior-art clearance claimed. These sources
screen two tempting formulations; they supply no new benchmark result or GO.

## 2026-09-16 C²RL-derived source lead

The independent [sltbaselines](https://github.com/ozgemercanoglu/sltbaselines)
repository acknowledges partial implementation sharing by C²RL's authors and
attributes its v2 contrastive kernel accordingly. Pinned source and enabled
training path now inspected; [audit and executable certificate](evidence/autonomous_search/C2RL_reimplementation_source_result.md).
This updates the available-source inventory without claiming an official C²RL
retrieval release, author-byte lineage or final-paper parity. The independent
project evaluates SLT; no translation score is promoted into the SLRet table.

Follow-up [retrieval task/resource gate](evidence/autonomous_search/C2RL_retrieval_resource_gate.md):
preprint §III-C compared with the pinned independent release's actual evaluators.
SLT generation and batch pretraining losses are not a full-gallery retrieval
evaluation. Public branch/tag/release/tree and linked initialization-resource
checks did not establish a trained C²RL retrieval pipeline. No final-paper parity,
global code-absence claim or new performance result. Source-only; no model assets
or benchmark TEST contents accessed. This completes the bounded two-turn lead.
