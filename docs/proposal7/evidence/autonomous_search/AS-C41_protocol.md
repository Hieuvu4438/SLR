## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (registered before inventory or outcome counts)
- Version Label: AS-C41-v1

## Natural one-word substitutions in both caption languages

Q09/layerD: AS-C16tested same-lexicon word-order changes, not real captions
differing by a single word. Census original7096TRAIN and519DEV separately;
no cross-split pairing, synthetic captions, new positives/negatives, training,
benchmark or changed task. Candidate rows are existing official paired samples.

Normalize both English caption_model and German caption_original with the
existing AS-C16words function (lowercase Unicode word regex, punctuation
excluded). Eligible English pairs have equal word counts and exactly one
position with a different word. Require the SAMErow pair also has exactly one
German word substitution, possibly at a different position. No insert/delete,
lemmatization, stopword/POS filtering, edit-distance or length threshold sweep.
All word types included: do not call them exclusively content-word contrasts.
Exclude identical complete deployed32token/segment/mask inputs from scoring;
report such exclusions. Record substitutions, positions, masked templates,
native/English text, IDs, and same-source/signer flags from existing forensics.
Require native forensics wording and IDs agree with manifest for every row.

Report English-only/German-only/intersection pair counts, distinct row counts,
joint masked-template counts and pair multiplicity. Text edit corroboration is
NOT translation alignment correctness, semantic opposition, sign gloss change,
or expert-validated signed-video minimal contrast. Source/signer matching flags
are descriptive, not linguistic control or causal independence. No expert
annotations are synthesized. Pair existence alone cannot support a method.

For eligible TRAINpairs, reuse pinnedR0frozen_train cache and AS-C16paired
scoring kernel for the four matrix cells(i,i),(i,j),(j,i),(j,j). Report four
directional margins and counts positive>1e-4,near|margin|<=1e-4,negative<−1e-4.
No full TRAINgallery score claim; numerical pair scoring may differ slightly
from blocked scoring. On eligibleDEVpairs validate paired-kernelR0four-cell
values against the existing full matrix at max absolute tolerance5e-5; report
zero validation pairs explicitly if none. No tolerance-selected pair omission.

For DEV use fixed original3seed score matrices and AS-C32uniform ensemble,
all parent hashes checked. Report four margins for every eligible pair and
fixed full-gallery ranks. For each model/direction, report number of original
persistent queries whose strongest incorrect candidate belongs to this pair
inventory, and how many have a margin<−1e-4. Report numerator/denominator and
indices; do not infer a broad failure from a tiny pair-selected subset.

No p-value or independent-pair significance claim: templates, repetitions and
shared clips induce dependence. No candidate generated solely from pair counts.
A measured pair failure still requires interpretation/provenance before any
mechanism claim. Empty/few pairs or uniformly separated pairs rejects only
this narrowly defined explanatory lead, not linguistic contrast sensitivity
generally. No ELSClexical method or native-caption preservation revival.

## Execution

    PYTHONPATH=shared:. timeout 300 /home/haipd/miniconda3/bin/python -m methods.information_probe.single_word_contrast_audit

Output AS-C41-SINGLE-WORD_run.json, no overwrite. Expected seconds to tens of
seconds, CPUinventory/small GPUscoring,300s hard timeout,progress per split,
monitor30–60s. Unit tests cover one-edit versus insert/delete/order/two-edits,
repeated identical sentences, positions and bilingual intersection. Manifest,
cache,source/forensics/checkpoint/score hashes retained. No encoder update,
test access or upload. Full11/11fallacy scan after results; no silent retry.
