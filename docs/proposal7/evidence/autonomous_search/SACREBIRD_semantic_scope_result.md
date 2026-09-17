# Human TRAIN audit: what the annotations can identify

2026-09-16. ANALYZED, AI-assisted academic-research-suite source verification.
Follow-up to the [exact join](SACREBIRD_train_join_result.md), under the separate
[comment-reading protocol](SACREBIRD_semantic_scope_protocol.md). The previous
turn made progress by establishing a previously unverified annotation resource.

## Reading result

All307TRAIN comments were read, represented by150distinct exact comment strings
with their complete source-row memberships. Repeated strings were read once;
whitespace-distinct strings were not merged. This is AI reading of existing
annotations, not a new annotation dataset or an independent DGS review.

The comments describe omissions, more-specific locations, lexical/numeric
differences, possible paraphrases, uncertainty and occasional dependence on a
previous sentence. Some comments contain questions or uncertainty markers. The
frequent literal `x` is not independently certified as an error-free label.
No new category counts, per-example labels, clean subset or training weights were
created from these observations. The source's four original flag columns remain
unchanged and retain their original meaning.

The primary paper's §3 confirms that TRAIN review compares glosses with German
transcriptions, using four TER-inspired categories. It explicitly excludes the
reordering category because German and DGS word orders differ. The separate
video-based back-translation procedure concerns TEST, not the TRAIN flags read
here. That methodological distinction is not evidence about TEST results.
[Czehmann et al., methods](https://www.sign-lang.uni-hamburg.de/lrec/pub/26064.pdf).

## Fixed first-comment check

Per protocol, inspect the first substantive comment, without choosing a model
error or a favorable discrepancy. Exact identifier:
`11August_2010_Wednesday_tagesschau-1`.

The existing official TRAIN gloss contains `FEBRUAR`; the official German
translation contains `august`; the local English model caption contains `August`.
This corroborates the reviewer's stated **gloss–text discrepancy** using the
actual local inputs. It does not establish a new error introduced by English
translation in this example. Conversely, agreement between German and English
does not verify either against the signed video. The calendar month in the
filename is not independent evidence of which month was signed.

No footage, pretrained features, hidden representations, scores or model outputs
were inspected. Thus we cannot decide whether the gloss, text, both, or neither
adequately captures the full signed message. We cannot label this pair as an
incorrect positive or derive a trusted February/August hard negative.
[Pinned author annotation file](https://github.com/DFKI-SignLanguage/sacre-bird-phoenix/blob/012b11c22b1c64b79325ec6b004b81693db86187/train_annotations_sacrebirdphoenix.csv).

## Identification boundary

Let G be the gloss statement, T the German statement, E its existing English
rendering, and S the signed meaning. Observing G≠T and E agreeing with T permits
at least two explanations consistent with this reading:

| Unobserved signed meaning | Interpretation consistent with the observed records |
|---|---|
| S agrees with G | The spoken-text supervision may mismatch the signed content. |
| S agrees with T | The gloss transcription may be wrong while retrieval supervision is appropriate. |

The annotation and the inspected triple do not choose between these explanations.
This is an identification argument about the available observations, not a claim
that both worlds are equally likely or that signed meaning has only two options.
It explains why correlating the human flag with TRAIN retrieval loss cannot by
itself establish visual information loss: the outcome needed for that attribution
is not identified by the flag. High or low TRAIN fit is compatible with several
annotation and model mechanisms.

Similarly, comments noting prior-sentence reference are evidence of an annotator's
interpretation of the gloss/text pair, not a verified original-recording join or
proof that an isolated signed video is uninterpretable. They do not repair Q31's
provenance gap. The explicit exclusion of reordering also prevents upgrading
unflagged rows into certified order-sensitive controls for Q06/Q09.

## Decision

The resource is useful for documenting gloss/text disagreement and constraining
claims made from gloss annotations. It does not supply the missing signed-content
labels for a new semantic model diagnostic. In particular:

- Do not treat the33lexical flags as33verified visual–English contradictions.
- Do not change positives, filter examples, mine counterfactual captions, create
  a clean benchmark subset, or relaunch reliability/soft-positive/lexical methods.
- Do not open the TEST back-translations to obtain the missing labels.
- Do not run a TRAIN-loss correlation merely because it is computable; no
  distinct admissible intervention would be identified by either result here.

Both resource and semantic-scope passes are complete. Move to another causal
layer; repeating the join, broadening comment categories, or inspecting additional
favorable examples would not resolve the attribution gap. This is a bounded
failure of this proposed use, not a global absence of compatible annotations or
a research barrier. No new candidate/Q38/Proposal8/GO. Goal remains active.

## Provenance, coverage and limits

Author repository pin `012b11c22b1c64b79325ec6b004b81693db86187`; TRAIN CSV hash
`004dc23564ba056ec76d06573ec4045d6a9c1bb050b7b3e4821756664b01d771` checked before
comment reading. Source in memory only; raw comment corpus not saved locally.
Official TRAIN CSV hash
`cc3dc2461f0a222b92f3927c24ac21c1467f3e5428b406ee7fe40bca1b0b8d44`.
English comes from the existing TRAIN manifest documented in the prior join.

Primary PDF hash
`58191697a49192d8e6d3b1c35dadbc1484e54ae13ee32405559b39e45164bfa2` checked before
extracting §3 via PyMuPDF. Section-based locator only; no local page claim.
§3's TEST-method descriptions were incidentally included with the TRAIN codebook;
no TEST results/examples or separate TEST CSV contents were read this pass.
Previous §4.1 sampling limitations remain: structured blocks, one reviewer,
no independently established inter-rater reliability. No full-paper reading,
video-semantic confirmation, new linguistic truth or efficacy claim.

No DEV scoring, TEST data, SEDS assets, model/checkpoint access, GPU, training,
upstream modification or external-model upload. Original annotation terms remain
CC BY-NC-SA4.0 plus separate corpus terms; only a minimal attributed example and
analysis are recorded. ARS verification prevented source disagreements from
being promoted into visual error labels. Venue/retraction/COI checks were not
expanded beyond the prior bibliographic verification.
