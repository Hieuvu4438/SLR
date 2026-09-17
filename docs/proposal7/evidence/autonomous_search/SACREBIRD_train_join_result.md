# Existing human annotation resource: verified TRAIN join

2026-09-16. ANALYZED; AI-assisted academic-research-suite source verification.
The preceding visual-mask turn was progress: its certificate rejected a concrete
contamination hypothesis. This turn changes layer to existing annotation evidence.

## New resource, bounded meaning

The author-hosted [sacre-bird-phoenix repository](https://github.com/DFKI-SignLanguage/sacre-bird-phoenix)
provides a TRAIN gloss/German-text audit. This is an existing annotation release,
not annotations created by this project. Its flags concern omissions and
discrepancies between glosses and German text; they are **not** video-grounded
relevance labels, nonmanual labels, or assessments of the English model captions.
Unflagged does not mean error-free. The repository also contains TEST resources,
which were not downloaded or opened.

Repository pin: `012b11c22b1c64b79325ec6b004b81693db86187`.
Allowlisted file: `train_annotations_sacrebirdphoenix.csv` only.
SHA256: `004dc23564ba056ec76d06573ec4045d6a9c1bb050b7b3e4821756664b01d771`.
The derivative release specifies CC BY-NC-SA4.0 and separate original-corpus
terms; no raw annotations or comments are redistributed here.

## Executed exact join

[Protocol](SACREBIRD_train_join_protocol.md),
[script](../../../../methods/information_probe/sacrebird_train_join.py),
[machine record](SACREBIRD-TRAIN-JOIN.json).

The check completed exit0 in.51s. It fetched the pinned TRAIN CSV in memory and
compared identifiers against existing TRAIN/DEV manifests, with no fuzzy mapping.

| Check | Result |
|---|---:|
| Rows / unique identifiers | 307 / 307 |
| Exact matches to existing7096-row TRAIN manifest | 307 |
| DEV matches / non-TRAIN identifiers / duplicate IDs | 0 / 0 / 0 |
| Missing category values | 0 |
| Gloss-omission flag1 | 97 |
| German-text-omission flag1 | 13 |
| Lexical-error flag1 | 33 |
| Minor-difference flag1 | 165 |

These are independently counted raw category values, not an inferred semantic
error rate. Categories can overlap; do not add them to count affected examples.
All category values were literal0/1. Comment strings were parsed as CSV content
but not semantically inspected, printed or retained. No captions or model scores
were compared. Manifest hashes are recorded in the JSON and match the existing
TRAIN/DEV provenance identifiers. No raw benchmark TEST file was read.

## Primary methodology check

Czehmann, V., Yazdani, S., Hamidullah, Y., Nunnari, F., & Avramidis, E. (2026).
[*“A Sacred Bird Called the Phoenix”. Auditing the most-used Parallel Corpus for
German Sign Language Recognition and Translation*](https://www.sign-lang.uni-hamburg.de/lrec/pub/26064.html),
LREC sign-language workshop. Metadata corroborated by
[the author institution](https://www-live.dfki.de/web/forschung/projekte-publikationen/publikation/17257)
and ACL author listings.

Section4.1 states that one deaf fluent L2 DGS signer annotated307gloss/text pairs
over25person-hours. Sampling starts at the beginning, takes blocks of10–11
consecutive items at intervals100through item1500, then intervals500to the end,
covering material associated with all nine signers. This is a structured sample,
not a simple random sample. The inspected section does not establish independent
multiple-rater agreement. Neither representativeness nor error-free controls can
be assumed. No corpus-wide prevalence or confidence interval is reported here.

Reading scope: repository README schema/limitations/license, primary publication
metadata and §4.1 in full. Not a full-paper review. Browser access to the workshop
record/PDF failed; direct read-only HTTP succeeded. Initial pdftotext extraction
failed because the executable was absent; PyMuPDF fallback succeeded in memory.
PDF SHA256 `58191697a49192d8e6d3b1c35dadbc1484e54ae13ee32405559b39e45164bfa2`.
Only section-based locators are used, not unverified local PDF page anchors.
PDF heading-location snippets and search abstracts incidentally mentioned the
separate TEST analysis; its results were not used to generate this TRAIN question,
select examples or infer model behavior. No TEST annotation/back-translation CSV
was accessed. The initial lead came from the TRAIN schema and identifiers.

## Decision and updated research boundary

This changes one resource prerequisite: an exact-join human gloss/text consistency
audit is now verified. It does **not** resolve Q11's cross-candidate relevance,
Q17's manual/nonmanual cue attribution, Q23's irreducibility, or the missing DEV
annotation support. It also does not license relabeling, filtering or changing
positive pairs. Those conclusions require different evidence.

The next justified use is a **TRAIN-only semantic scope check**: inspect the
existing comment/codebook meaning and determine whether the annotations locate a
specific distinction relevant to the current visual–English task, or merely
document gloss–German discrepancies. No new semantic labels should be invented.
Only if a distinct admissible decision follows should a fixed model diagnostic
be registered. A correlation between these flags and supervised TRAIN loss alone
would not establish visual information loss or a generalization mechanism.

Generic noisy-label reweighting, soft positives, lexical repair, teacher mining
and grouped-risk methods remain closed. Do not use this resource to reopen them
or to declare a benchmark ceiling. No architecture/training candidate, Q38,
Proposal8, GO or global barrier. Goal remains active.

## Evidence grade and limits

Strong computational evidence for exact identifier compatibility; one descriptive
human audit (field-relative LevelVI) for the category judgments, not independent
replication of signed meaning. Annotator/method information is author-reported.
Venue metadata checked, not retraction databases, indexing or complete COI
disclosures. No SEDS assets, feature/model/checkpoint access, GPU, training,
new corpus, new split, changed label or benchmark modification. ARS verification
kept the resource discovery separate from a claim of semantic model failure.
