# PH multi-stream alignment: not an independent cue-validation target

2026-09-16. Academic-research-suite / deep-research, fact-check inline.
AI-assisted source inspection; no model run or linguistic annotation.

## Decision

Do not acquire the alignment archives to serve as independent ground truth for
Q17/Q01. A published weak-label pipeline is verified, but neither a human timing
certificate nor an exact archive-to-current-TRAIN join is established. Calling
an archive “alignment” does not make it manual annotation. This rejects the
proposed validation use, not the utility or accuracy of weak supervision.

The previous goal turn was **no progress**: a source-repair training question
was asked without new evidence. This continuation did not answer that question;
it is not authorization to reopen closed families. This pass instead checks a
distinct resource prerequisite within existing scope.

## New primary evidence

The official [RWTH directory](https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/)
lists two relevant PH2014T archives: a2018train_val/3stream alignment archive
(6.5M as displayed) and `phoenix2014T-multi-stream-alignments.tar.gz` from2019
(3.8M). These are directory metadata, not verified content or split inventories.
No archive bytes, learned features or models were downloaded.

[Koller et al., author-hosted PAMI preprint](https://www-i6.informatik.rwth-aachen.de/publications/download/1099/Koller-PAMI-2019.pdf):
§3 estimates frame alignment through iterative CNN-LSTM-HMM/Viterbi training.
§4.1 builds weak mouthing alternatives from German/gloss alignment and a
pronunciation lexicon, with truncated forms and a non-mouthing alternative.
§4.2 derives candidate handshape sequences from a dictionary and explicitly
describes them as weak, not manually refined for each dataset instance.
This establishes the published supervision provenance, not exact archive parity.
Read scope: §3 opening and §§4–4.2, with additional method locators; not full paper
or a reproduced result. Browser search contexts incidentally included published
benchmark tables; none supplies a hypothesis, selection decision or efficacy claim.

## Pinned source path

The author's homepage links the [Re-Sign repository](https://github.com/huerlima/Re-Sign-Re-Aligned-End-to-End-Sequence-Modelling-with-Deep-Recurrent-CNN-HMMs).
HEAD resolved on2026-09-16 to `b1c7981ed51aace2c8c84a9ddf30d06500509f1c`.
Full README and three explicit source files were fetched read-only at that pin.
The tree query was untruncated; only filenames under the two relevant code
directories were printed. Dataset/cache/log contents were never fetched.

- `01.generate-frameLabels/doit_fromalign.sh` derives TRAIN/dev frame-label
  outputs from the preceding iteration's alignment cache, lexicon and allophones.
- `06.align-2stream/alignment.sh` selects TRAIN posteriors and corpus paths for
  alignment. `config/alignment.config` supplies corpus/lexicon, HMM state and
  transition settings, and a hybrid scorer. These are model-based inputs.
- This checked source has configuration placeholders and a `dry` action default;
  it is not a verified runnable reproduction or certificate for either archive.

SHA256:

| Inspected file | Hash |
|---|---|
| README.md | `98db5f3cecd1faa24c3c978f06609f89f81e61f840d027c77ed5ea8591df6f53` |
| doit_fromalign.sh | `e6c12f864cc25cf9cb1e0208672561fc959bc510370d782ee7ed6440b4400f28` |
| alignment.sh | `6e46c54974c47b8d6f98d4a2b2db7eceaf7cc650a0c43f756a2cc23ce7537ec7` |
| alignment.config | `f7de1fc7c9a4a05aeb48f8d99d64b256b38f415365e5691e14a1544fe5fc44d8` |

## Why this does not calibrate the proposed probe

Project inference: let A be an automatic alignment constructed using visual
input X, supplied gloss/text L and learned parameters. Agreement of a probe with
A measures prediction of this constructed target. It does not independently
verify the actual cue C, its temporal boundaries or its retrieval relevance.
In particular, conditioning A on L permits linguistic information to enter the
target; a good score against A alone cannot separate cue recovery from lexical
correlation. This is not proof that A ignores video, that A is wrong, or that
all comparisons to pseudo-labels are invalid. It is an identification limitation
for the specific claim we wanted to test.

Do not convert these resources into a new teacher/support/local-margin model,
claim manual timing, or reuse their DEV pseudo-labels for training. No archive
download merely to get a larger label count, new corpus join campaign, or generic
probe training follows. Complex-notation and alignment-provenance passes now
complete this annotation-resource layer; change layer next.

## Verification limits

The directory, paper and code belong to the same author/release lineage, not
independent empirical corroborations. Directory/README/code are descriptive
LevelVI sources, high fitness for their own stated contents; paper methods are
primary computational evidence, not replicated retrieval efficacy. No independent
COI, retraction, DOI or full publication-version audit claimed. No human-read
attestation. Primary searches used the two exact archive-name strings.

No dataset examples, TRAIN/DEV/TEST CSVs, alignment members, videos, feature arrays,
checkpoints, SEDS assets, training or GPU access in this pass. Source scripts
contain TEST path/copy strings but were not executed. The scope question from
the previous turn is not a global research blocker: this permitted check was
available and has now been completed. No method GO, Proposal8 or global barrier.
