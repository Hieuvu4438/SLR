# Draft video-relevance diagnostic — not an approved human study

**DECLINED — 2026-09-22:** the user explicitly requested continuing without
human annotation. This branch is stopped. No sampling manifest was generated,
no footage accessed and no judgments collected. Retained only as decision history;
do not recruit reviewers, render a packet or substitute AI-generated judgments.

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan, using existing research-question and residual-analysis inputs
- Origin Date: 2026-09-22
- Verification Status: UNVERIFIED (no judgments collected)
- Version Label: relevance_review_protocol_v1

## Purpose and scope

Question: among selected persistent PH DEV confusers, which distinctions are
actually recoverable from the signed video without outside context?

This is bottleneck discovery for Q01/Q23, not a new retrieval method or a changed
benchmark. The existing six-candidate screen has no supported survivor. Scores,
gloss disagreements and spoken-text comparisons do not establish signed meaning.
Two competing explanations remain: a visually distinguishable model error, or a
match the official singleton relation does not adequately characterize.

Preparation is local and metadata-only. No reviewer has been recruited, no
footage distributed, no institutional approval/exemption claimed, and no new
semantic labels created. The user has been asked whether qualified DGS review or
compatible existing video-level judgments can be provided. Only preparation may
proceed before that answer and the applicable governance decisions.

Historical boundaries remain binding: do not reuse the307 TRAIN gloss/German
comments as visual truth, acquire the incompatible mouthing archive, inspect TEST
back-translations, mine fuzzy-gloss negatives, or revive relevance-weighted losses.
This proposal requests new independent evidence; it does not silently authorize it.

## Sampling contract, fixed before case selection

Input: the existing `gcn_residuals_20260922.json` record, whose selected checkpoints
are seed42/666, seed1337/222 and seed2026/222. Verify its referenced official DEV
CSV and three fusion metric hashes before selection; no active job or features.
Selection uses recorded ranks and IDs, not a new inference pass.

Eligible persistent rows: error in all three seeds, the same top non-paired
candidate in all three, and a strictly negative paired-minus-confuser margin
in every seed. Eligible model-correct controls: paired candidate rank0 and a
strictly positive margin in all three, with stable top non-paired candidate.
Controls are **model-correct**, not human-verified semantic controls.

Draft quota:12 persistent T2V,12 persistent V2T,3 model-correct T2V and3
model-correct V2T trials. Select by SHA256 of a fixed seed/stratum/query-ID key.
Use distinct query IDs across all30 trials and at most one query broadcast
source per stratum. Broadcast source is the ID with its final segment suffix
removed; this is a sampling heuristic, not a certified recording-time join.
If a quota cannot be met, fail rather than silently relaxing it.

Add four exact-content repeat trials, one from each stratum, with A/B positions
reversed; randomize display order by a separate fixed hash. Repeats assess
within-reviewer consistency, not extra independent observations. Duplicate
candidate clips across trials may still occur and must be recorded. Allocation
is exploratory and source-diverse, not representative random prevalence sampling.
Do not generalize sample percentages to all113/98 persistent errors.

The generated manifest is **internal and unblinded**: it retains source IDs,
model-selection strata and the official paired position for audit. It must never
be given directly to reviewers. Actual stimulus rendering is deferred; filenames,
gloss labels, official-pair markers, scores and error/control strata must be hidden
in a separate reviewer interface before data collection.

## Reviewers and instruments — proposed, not recruited

Prefer two independent adults fluent in DGS and written German, familiar with
weather-domain signing. The user/institution must determine qualifications,
compensation, consent, availability and any needed ethics review. A single reviewer
can supply exploratory observations but not inter-reviewer agreement. No LLM
substitute, automatic visual gloss interpretation or fabricated consensus.

T2V trial: one native German query and two anonymized sentence clips, official
paired versus stable confuser in randomized positions. V2T trial: one sentence
clip and two native German captions in randomized positions. These judgments
address video/native-text relevance; the retriever's English input is a separate
potential translation confound, not automatically verified by this design.

Judge each candidate independently first:

- adequate match to the signed content;
- partly matching but missing or contradicting important content;
- incompatible;
- cannot determine from the available clip.

Then compare: A better / B better / equally acceptable / neither acceptable /
cannot determine. Do not force a unique answer. Record confidence (low/medium/high),
the decisive visible cue with approximate time span, and whether earlier discourse,
clip boundaries, occlusion or domain knowledge prevent judgment. Free-text cue
descriptions precede optional categories; do not prime reviewers toward CTC/order,
handshape or nonmanual explanations. Native-language instructions need expert
checking before use. This instrument has not been validated.

## Collection and governance gates

1. User confirms an available reviewer or identifies existing compatible judgments.
2. Responsible institution/user determines approval or exemption requirements;
   status is **not assessed**, never assumed exempt because corpus videos are public.
3. Approve accessible consent, voluntary withdrawal, workload and compensation;
   define who sees reviewer identities, responses and corpus footage.
4. Verify corpus use/sharing terms, access-controlled storage, retention/deletion
   period and secure transfer. The repository is not presumed approved response
   storage; do not place identifiable reviewer data here.
5. Verify exact DEV clip IDs, normal playback and full frame extent; no soundtrack
   or metadata may reveal a hidden answer. Confirm which visual context the actual
   model sees. No original-broadcast retrieval is authorized by this plan.
6. Create and test the blinded interface; independently verify orientation/key
   mapping. Reviewers work independently before any adjudication discussion.

The academic skill permits planning but keeps recruitment and collection behind
these unresolved conditions. No participant messages, uploads or scheduling occur.

## Analysis and decision use

Report raw counts by direction and selection stratum, missing/uncertain judgments,
per-reviewer results, repeat agreement and inter-reviewer disagreements. No p-value,
power or prevalence precision is promised for this small purposive pilot. Do not
discard disagreements or uncertainty to manufacture a clean benchmark. Retain
original independent responses separately from any later adjudication.

If independent reviewers identify repeatable visible distinctions across at least
three distinct query broadcast sources, inspect those distinctions against the
actual model interface and blacklist. This is an exploratory trigger for mechanism
formulation, **not** an efficacy threshold or SUPPORTED-FOR-PILOT decision. Agreement
that both candidates fit motivates only a relevance limitation statement; it does
not authorize relabeling official positives or training soft-positive objectives.
Uncertain/low-agreement results do not establish either explanation.

No semantic findings or trained method can be claimed before the missing review
exists. No official R@K is recomputed with human labels. Any future mechanism still
needs separating controls, matched pilot gains, multi-seed and second-dataset
confirmation, and a locked TEST comparison under the original protocol.

## Limitations and stopping rule

DEV and the three checkpoints have already informed selection; this is discovery,
not confirmatory evaluation. Source-diverse quotas exclude some persistent errors.
Matched-looking controls are not ground truth. Clips may be recognized despite
blinding. German-caption review does not validate English tokenization. Reviewer
agreement cannot by itself identify which neural interface loses a cue.

If qualified review/compatible judgments are unavailable or declined, stop this
annotation-dependent branch. Do not repeat historical annotation lookups, produce
model-generated labels, or claim that human review is a universal prerequisite
for all possible SLRet research. The unresolved method goal remains unchanged.
