# Research brief — compositional sentence-level sign retrieval

## Primary question

Can a SEDS-independent multi-vector model that preserves contextual sign-window
and text-token representations improve official bidirectional Recall@1,
especially for compositionally novel queries, over a matched single-vector
encoder on PHOENIX-2014T without changing relevance labels or selecting on TEST?

## FINER assessment

| Criterion | Score | Reason |
|---|---:|---|
| Feasible | 4/5 | PH annotations, raw videos, public checkpoints, a verified SignRep checkpoint, and cached TRAIN512/DEV519 window features are local. Full replication needs more compute. |
| Interesting | 5/5 | Current systems compress a visually and linguistically structured sequence into a global score while the measured eval captions are usually new compositions of familiar lexical material. |
| Novel | 4/5 | Late interaction exists in generic video retrieval and token alignment exists in SLT, but the screened work does not establish a sentence-level SLRet architecture that retrieves directly from sign-prior window banks with dual-level late interaction. This remains a bounded, not exhaustive, novelty judgment. |
| Ethical | 5/5 | Existing released research datasets only; no new human-subject collection or identity inference. Dataset and language limitations must be disclosed. |
| Relevant | 5/5 | It targets official SLRet quality and separately measures whether gains transfer beyond familiar sentence/source structure. |
| **Average** | **4.6/5** | Meets the progression threshold. |

## Scope

In scope: sentence-level T2V and V2T; PH first; official one-to-one relevance and
gallery unchanged; DEV-only selection; matched pooled control; CSL-Daily and
How2Sign as confirmation datasets if the PH mechanism survives; accuracy,
robustness slice, and retrieval cost reported separately.

Out of scope in the first tranche: new TEST labels, multi-positive evaluation,
hard-negative mining, optimal-transport/dustbin objectives, source/signer gates,
reranking or ensembles, SEDS fusion changes, and a full foundation-model pretrain.

Assumptions: released captions are an imperfect spoken-language view of signing;
simple lexical units measure surface compositional support rather than sign
semantics; filename-derived PH source groups are not independently verified
recording identities.

## Measured dataset structure

All values come from the hashed released annotations and the reproducible script
in this directory.  No model predictions or TEST-driven selection were used.

| Dataset/split | Key observation |
|---|---|
| PH DEV | 477/519 rows use only TRAIN-seen word units, but only 161/519 have all adjacent bigrams seen and only 25/519 repeat a complete TRAIN caption. Under the frozen operational definition, 316/519 are compositionally novel. |
| PH TEST | 593/642 have all word units seen, 229/642 all bigrams seen, 32/642 exact captions seen, and 364/642 meet the compositional definition. All 642 filename-derived source groups occur in TRAIN. |
| CSL DEV | 1,028/1,077 have all character units seen but only 63/1,077 all bigrams; 965/1,077 meet the compositional definition. TRAIN and DEV semantic IDs are disjoint. |
| CSL TEST | 1,051/1,176 meet the compositional definition against TRAIN. Separately, 1,173/1,176 TEST rows belong to 795 semantic/caption groups also present in DEV, while no semantic IDs overlap TRAIN. |
| How2Sign DEV | 1,268/1,741 use only TRAIN-seen word units, 250/1,741 have all bigrams seen, 47 repeat complete TRAIN captions, and 1,018/1,741 meet the compositional definition. TRAIN/DEV source-video IDs are disjoint. |

Operational compositional definition: every unigram occurs in TRAIN, at least
one adjacent bigram is unseen, and the complete normalized caption is unseen.
This does not prove that the corresponding signs are visually novel; it is a
predeclared diagnostic slice, never a replacement for official R@K.

## Subquestions and inherited bindings

1. Does direct late interaction beat a parameter- and exposure-matched pooled
   encoder? Inherits PH, official labels, DEV selection, same cached inputs.
2. Is any improvement concentrated on the compositional slice or shared across
   exact-seen, lexical-novel, length, and source slices? Inherits the same frozen
   model and gallery; no slice-specific tuning.
3. Does the mechanism transfer to a second language/dataset and remain practical?
   Inherits unchanged relevance rules; dataset-specific tokenization is declared,
   and cost is reported independently.

## Devil's-advocate checkpoint 1

Verdict: PASS WITH MAJOR CAUTION.

The strongest counterargument is that lexical novelty in translations is not
visual/sign compositional novelty, and late interaction may merely exploit extra
capacity or public-pretraining scale.  The design therefore requires a pooled
control with the same inputs/projectors, a single-level ablation, explicit public
pretraining disclosure, and official full-gallery metrics.  A slice-only win or a
gain caused by changing positives is not success.
