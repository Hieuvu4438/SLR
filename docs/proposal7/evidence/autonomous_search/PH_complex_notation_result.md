# Complex PH TRAIN labels: retained coarse tags, absent detailed postfixes

## Material Passport

2026-09-16; academic-research-suite / deep-research, inline source verification.
ANALYZED existing annotation resource, not a new annotation set or model result.

## What changed

The earlier POST_C44 census verified the complex TRAIN file's membership and
missing timestamps, but explicitly did not interpret its notation. This pass
establishes which literal annotation markers are present. It would be incorrect
to say that PH2014T has no existing nonmanual-related TRAIN annotation signals.
It would also be incorrect to equate this file with detailed simultaneous-hand
or viseme ground truth. This distinction narrows Q17's missing resource claim.

No candidate or experiment is promoted. We have not established a cue-isolating
contrast, anatomical localization, annotation completeness, representation loss,
or relevance to any retrieval error. Positive tag counts alone cannot supply
the missing causal test or override the closed lexical/RCLI/stream families.

## Primary-source scope

The official [2012 annotation conventions](https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/annotation-convention_20120820.pdf)
were read completely (two pages, §§1–4 and table; browser extraction and page2
screenshot requested). The guide distinguishes head-shake `neg-` from manual
alpha negation. It defines detailed postfix notation for mouthing, facial
expression and left-hand content, and special nonmanual/holding signs. It also
states legal restrictions on distributing original ELAN files. This documents
the original scheme, not automatic preservation of every field in PH2014T.

Local release README explicitly identifies the complex annotation and says the
recognition evaluator simplifies it. The evaluation shell source removes special
tokens, localization/classifier prefixes and repetition suffixes. It was read,
not executed; no recognition ground truth, hypothesis, DEV or TEST file opened.
The script's transformations are not a proof that all source strings map exactly
to the standard CSV. No exact full normalization-parity claim is made.

## Census

[Protocol](PH_complex_notation_protocol.md),
[code](../../../../methods/information_probe/complex_notation_census.py),
[complete counts and hashes](PH-COMPLEX-NOTATION.json).

Each CSV has7,096unique IDs exactly matching the TRAIN manifest. Standard/complex
token totals55,247/67,781; unique token counts1,085/1,232. All start/end fields
remain−1. Below, counts are literal syntax occurrences, not inferred visual events.

| Marker | Standard tokens / rows | Complex tokens / rows |
|---|---:|---:|
| `neg-` prefix | 175 / 132 | 175 / 132 |
| `negalp-` prefix | 48 / 47 | 48 / 47 |
| `poss-` prefix | 138 / 127 | 138 / 127 |
| `loc-` prefix | 0 / 0 | 2,242 / 1,665 |
| `cl-` prefix | 0 / 0 | 756 / 521 |
| `-PLUSPLUS` suffix | 0 / 0 | 688 / 632 |
| `__EMP__` | 0 / 0 | 304 / 285 |
| `__LEFTHAND__` | 0 / 0 | 442 / 312 |
| `__HOLD__` | 0 / 0 | 679 / 610 |
| `__PU__` | 0 / 0 | 897 / 786 |
| `__ON__` | 0 / 0 | 2,560 / 2,465 |
| `__OFF__` | 0 / 0 | 2,753 / 2,663 |

Both files have zero literal `mb:`, `mk:`, `lh:`, `name:`, `time:`, `loc:`, `obj:`,
`in:` fields, zero parentheses/hash variants, and zero `lh-`, `bh-`, `negalpha-`
prefixes. The guide uses both an abbreviated alpha-prefix example and a longer
spelling in prose, hence both forms were counted separately, without conflation.
Standard has52literal-plus tokens in48rows; complex has none. The complete JSON
also preserves every distinct `__??...__` marker count. They are not converted
into cleaned labels or certified annotation errors.

Execution exit0 in0.106625645s. Assertions verify counts/uniqueness, exact TRAIN
join and split. Source and script hashes retained. No independent linguistic
validation or statistical performance test; this is a complete fixed-file census.
No token identities, raw captions or per-example cue labels were exported.

## What this resource can and cannot support

- Existing head-shake-negation notation is a possible positive annotation signal;
  its aggregate count is already present in standard glosses. It is not newly
  recovered supervision and does not alone prove that absence means no head shake.
- Detailed original postfixes are absent in these files. Broad special markers
  do not specify which viseme, facial configuration, held-hand sign, or frame
  interval occurred. The transformation from original annotations to each special
  marker has not been reconstructed; do not infer an anatomical mask from its name.
- The retained localization/classifier tags are descriptive sign-category data,
  not a release of signing-space coordinates or a direct relation-level control.
- Equal aggregate counts across standard/complex files do not prove per-token or
  per-row alignment. No alignment model, normalized-gloss pairing, contrast labels
  or fuzzy-gloss rescue was constructed.
- These markers establish annotation availability, not statistical independence
  from lexical content, signer, sentence length or caption identity. A successful
  sentence-tag probe could use those correlates without recovering the named cue.
  A failed probe would likewise not establish absent visual information.

For now this supports descriptive annotation analysis, not a calibrated test of
rank-critical information loss. Do not train another generic probe from these
counts, reinterpret C44/C45, mine polarity pairs, or introduce extra supervision
into a supposedly matched gloss-free method. A future diagnostic must specify
what existing evidence isolates the cue and validates its retrieval consequence;
that prerequisite is not met here. Detailed-postfix/viseme acquisition from this
file is ruled out; broader nonmanual annotations are not globally ruled out.

## Integrity and next action

The previous CMCM counterexample was progress, not GO. This turn changes the
annotation-resource assessment, not a retrieval score. No trained model/features,
raw video, DEV/TEST data, SEDS resource, new positive set, benchmark, label or
training run. Directory discovery exposed only evaluation filenames, not their
contents. A few pre-protocol TRAIN lines were read to identify file syntax;
the protocol fixed the complete census before its execution.

ARS source discipline prevented treating an original codebook as proof that its
detailed fields survive the release. No human-read attestation or independent
linguistic reviewer is claimed. No Q38, Proposal8, method GO or global barrier.
Leave the goal active; this resource alone does not justify another model run.
