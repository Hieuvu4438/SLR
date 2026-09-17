# Post-C44: the published mouthing labels are not a direct PH2014T join

## Material Passport

academic-research-suite / deep-research, fact-check; 2026-09-15. ANALYZED.
AI-assisted source inspection and metadata census, not a linguistic annotation,
method pilot, retrieval result or footage-overlap certificate. Previous goal turn
was progress: AS-C44 corrected a probe's scope. This turn resolves the ensuing
resource-compatibility question with newly inspected release documentation.

## Finding that changes the next action

The previously unavailable web README exists in the local release. Its
[split guarantees](/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/README:36)
state that the ECCV2014 manually annotated mouthing sequences were excluded from
**all** PH2014T sets. It also excludes PH2014 DEV/TEST segments from PH2014T TRAIN.
The [additional information](/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/README:43)
documents changed sentence boundaries and possible boundary errors.

Therefore the proposed direct join of those manual mouthing labels to our current
TRAIN/DEV clips is not supported by the release contract. No archive download or
filename-normalization experiment is justified for that join. The handshape-only
resource still does not supply joint hand–face contrasts, and PH2014 DEV origin
does not establish PH2014T DEV membership. Do not promote either to training data.
This is a documented exclusion, not an independent pixel-level remeasurement;
other manual resources and the wider simultaneity question are not ruled out.

The original dataset paper independently of our code describes boundary changes
and cross-release TRAIN versus DEV/TEST nonoverlap. Its authors overlap with the
README authors, so this is corroboration within the release lineage, not an
independent replication. Read scope: title/abstract and §4; no performance claim
is used. [Camgöz et al., Neural Sign Language Translation, CVPR 2018](https://www-i6.informatik.rwth-aachen.de/publications/download/1064/Camgoz-CVPR-2018.pdf).

## Local census

[Machine record](POST_C44-ANNOTATION-COMPATIBILITY.json), first execution exit0,
0.076262s. All seven input hashes, code hash and preregistered scope hash retained.

| Permitted annotation file | Rows / unique IDs | Manifest identity set | Start/end | Video references |
|---|---:|---|---|---|
| Standard TRAIN | 7,096 / 7,096 | Exact TRAIN match | All −1 / −1 | All sentence-local frame globs |
| Complex TRAIN | 7,096 / 7,096 | Exact TRAIN match | All −1 / −1 | All sentence-local frame globs |
| Standard DEV | 519 / 519 | Exact DEV match | All −1 / −1 | All sentence-local frame globs |

All schemas are name, video, start, end, speaker, orth, translation. These files
provide no populated start/end coordinate for cross-release frame linkage.
Standard TRAIN/DEV translations exactly match the corresponding native manifest
captions, 7,096/519. Complex TRAIN has zero *byte-exact* translation matches;
this is not a claim of semantic disagreement or an instruction to change captions.
The initial inspected examples include punctuation differences. No semantic
interpretation of complex gloss notation was used.

CSV parsing and an independent line/field parser agree on every field in all
three files, with explicit delimiter-width checks. TRAIN/DEV identities are
disjoint. No TEST file, annotation pickle/gzip, video, feature, checkpoint or
score was opened. No archive acquired, no fitting, no changed positives.

## Source verification and limitations

The local README (SHA b73534fcde717c95f24819f9cc33d8458f5cbbe4d136d058a389cfe7e87aa269)
is primary release documentation, descriptive Level VI, high fitness for its own
release contract but not empirical proof of actual overlap. The original CVPR
paper is primary computational research; §4 is descriptive Level VI evidence for
dataset construction. Both are used with an explicit same-lineage limitation.
No independent linguistic reviewer, broad retraction/indexing check or external
archive validation is claimed. Public directory listings only corroborate that
named resources are offered, not their contents.

Search terms included PH2014/PH2014T segmentation and handshape mapping, the exact
mouthing archive name plus annotation format, and the handshape archive name plus
README. CVF returned403 and an author thesis exceeded the browser's size limit;
an incorrect attempted author-PDF path did not resolve. Following the actual
author-homepage publication link retrieved the correct paper. No failed source
was treated as evidence. The earlier web README404 is now resolved by a hashed
local copy, not by inventing its content.

## Decision

Close the **direct published-mouthing-label join** route for this diagnostic.
Do not repair it with pseudo-labels, sentence-ID heuristics, a new benchmark or
new manual annotation. AS-C44's scope correction remains valid, but cannot launch
a generic stream/position/synchronization model or reopen RCLI. No AS-C45 has been
registered and `method_go=false`.

The next cycle must move away from this annotation-dependent route to a different
testable, open causal mechanism. No further handshape/mouthing archive acquisition
for this join is pending. The optional user annotation question is not a global
blocker. This result is concrete progress, not success or global exhaustion.
