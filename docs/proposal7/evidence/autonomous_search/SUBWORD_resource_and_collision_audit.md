# Subword identity in sentence retrieval: resource and prior-art gate

## Material Passport

- Origin Skill: academic-research-suite / deep-research; inline scoping and source verification
- Date: 2026-09-15
- Verification Status: ANALYZED; no model experiment or method GO
- Disclosure: AI-assisted research, not independent human linguistic review

## Decision

Do not launch a fingerspelling/character-branch experiment on the current evidence.
The scientific question is plausible, but a current-project failure signature is
unmeasured. The specific How2Sign human-gloss resource proposed for a diagnostic
is not verified available, and generic character modeling or fingerspelling
retrieval has direct prior art. This is a bounded prerequisite result, not a
universal rejection of sublexical information or an exhausted research space.

The preceding CiCo census turn was **progress**: it completed an exact exposure
measurement and changed the next action. This turn changes the research question
from pseudo-clip grouping to text/annotation-side subword identity; it does not
run another crop/NMS/teacher or covariance experiment.

## Research-question scoping

Primary question: **Can existing human annotations identify fingerspelling-related
failures of the current sentence-retrieval system without changing its official
task, gallery, or positives?**

Three questions considered, not three surviving method candidates:

| Question | FINER F/I/N/E/R (1–5) | Decision |
|---|---|---|
| Existing-label attribution of fingerspelling-related sentence-retrieval errors | 2/4/2/5/4 | Resource verification first; no measured error signature yet |
| Does swapping subword for character modeling fix the problem? | 3/3/1/5/4 | Not a novel mechanism by itself; known prior art, unmatched-model confound |
| Does lost high-frequency visual spelling information explain errors? | 1/4/2/5/4 | No validated local spelling events or adequate matched intervention; do not assume visual loss |

These are provisional prioritization judgments, not empirical results or user
ratings. The primary question averages 3.4, but its resource gate remains unmet.
The other two fail at least one minimum criterion. None passes method selection.

Sub-questions, all inheriting the primary scope (existing permitted TRAIN/DEV,
sentence T2V/V2T, unchanged positives, no TEST selection or new annotations):

1. Do published gloss conventions identify fingerspelling explicitly?
2. Are those annotations actually available with usable dataset/split provenance?
3. Would the obvious proposed intervention be distinct from existing work and
   closed lexical-support, extra-stream, teacher or acquisition mechanisms?

## Source verification findings

### How2Sign: annotation convention is not resource availability

[V] The [How2Sign paper, arXivv2](https://arxiv.org/pdf/2008.08143), appendix §7,
describes `FS:` as the marker for a fingerspelled word. §3.3 describes linguistic
annotation using ELAN. These establish an annotation convention and an author
account of collection, not a downloadable current TRAIN/DEV inventory. Its
§3.4 approximate fingerspelling statistic is not a measured prevalence among
our retrieval failures and is not used to set a diagnostic threshold.

[V] The [current official download page](https://how2sign.github.io/) lists RGB,
keypoints, English translations and re-aligned translations, but no gloss download
link in the inspected download section. Its header advertises gloss annotation
and its release section is still marked under construction. No archive link was
followed and no dataset was acquired.

[V/A] An [owner response dated 2022-10-14](https://github.com/how2sign/how2sign.github.io/issues/5#issuecomment-1278618505)
says the annotated subset was being cleaned and full gloss collection had been
suspended for lack of qualified annotators. This is the latest maintainer
statement found in the fetched thread, not a verified assertion about all private
or subsequently hosted resources. The 2025 [issue #24](https://github.com/how2sign/how2sign.github.io/issues/24)
is closed, but its two comments merely point to issue #5; closure is not proof
that annotation files were released. API author-association fields distinguish
the owner response from non-maintainer requests.

Result: a usable current-project fingerspelling annotation join is **unverified**.
Do not substitute capitalized English words, proper-noun detection, a `FS` string
heuristic in an unrelated gloss convention, or a model's prediction for human
evidence of a fingerspelling event. No absence-of-fingerspelling claim about PH
or How2Sign is made. No exhaustive filesystem absence certificate is claimed.

### Direct prior-art collisions

[V/A] Shi, Brentari, Shakhnarovich and Livescu,
[Searching for fingerspelled content in American Sign Language](https://aclanthology.org/2022.acl-long.119/),
ACL 2022, describes FSS-Net as joint fingerspelling detection and matching to
text. Its target is fingerspelled keywords/phrases in untrimmed ASL videos, not
our sentence-level paired gallery. Read scope: official metadata and abstract;
full PDF retrieval failed because the browser rejected its size. No code or
published performance reproduction is claimed. The abstract alone is sufficient
to reject a claim that adding fingerspelling-specific search is a new general
principle; it does not establish exact equivalence of an unformulated new method.

[V/A] Tanzer,
[Fingerspelling within Sign Language Translation](https://aclanthology.org/2025.naacl-long.19/),
NAACL 2025 (2024 preprint), evaluates character-level ByT5 modeling versus
subword-level modeling and mixing fingerspelling recognition training data into
sentence translation. The author reports gains from the former and mixed effects
from the latter. Read scope: official metadata/abstract and first-page primary
PDF search excerpt; not the full 80-page paper. Its FLEURS-ASL annotations and
translation task do not supply a PH/How2Sign sentence-retrieval diagnostic or a
matched CiCo intervention. No FLEURS data or external recognizer was downloaded.

The official ACL records establish the publications and their stated scope.
DOI resolver attempts were blocked by the browser; do not label the references
as DOI-resolution-verified or independently replicated.

### CiCo code: lack of an explicit character head is not a loss certificate

[V] Complete local `CiCo/CLCL/modules/tokenization_clip.py` inspected, at SLRT
revision 38a4f7b00da7a858d59b7fabe5093876a84db8e0, file SHA256
60af2eccd9d1208933213b57d24c9ac337ca4be935bc6550a7300b656a6f3bb5.
The tokenizer contains byte-to-symbol and inverse maps, a token-ID decoder,
and byte reconstruction in `decode`. It also performs text repair, HTML
unescaping, whitespace normalization and lowercasing before tokenization.

[I] Subword tokenization does not, by itself, prove that retained token IDs lack
spelling information. This source reading is not a bitwise round-trip certificate
for arbitrary raw text, a truncation audit, or proof that a trained embedding
exposes every character distinction. Representation accessibility would need a
controlled test with an actual relevant failure signature. Generic tokenizer,
context, truncation and lexical-support rescue sweeps remain unsupported.

## Source quality and access limits

Field-relative grading: the ACL papers are primary ML sources (controlled-study
claims, approximately Level III), provisionally B for their stated mechanisms,
not established transferable effects. The dataset paper is a primary descriptive
resource account (Level VI), suitable for its stated conventions. The website
and owner comment are primary release-status evidence (Level VI), not independent
replications of the paper or a comprehensive current availability proof.

Official ACL/CVPR identities and author lists are verified. Dedicated predatory-
venue databases, retraction searches, and full funding/conflict disclosures were
not audited; no misconduct or conflict inference is made from missing access.
How2Sign paper, site and maintainer are one provenance lineage. The two unrelated
method papers establish prior-art boundaries, not independent confirmation that
How2Sign's gloss resources are unavailable. Search absence is not novelty proof.

Searches included `sign language retrieval fingerspelling sentence retrieval
phonological`, `PHOENIX 2014T fingerspelling annotations gloss`, `How2Sign
fingerspelling annotation existing dataset`, and exact paper titles. Third-party
annotation-tool configurations, automatic summaries, unrelated datasets and
secondary survey tables were not used as annotation provenance. Failed access:
UPC-hosted PDF timeout; CVF supplement 403; FSS PDF size limit; arXiv HTML cache
miss; DOI/browser API safety rejection. Public GitHub API reads recovered the
relevant maintainer comments without authenticated access or external writes.

## Incidental TEST-information disclosure

Fetching the public issue #5 comments unexpectedly returned an unrelated
non-maintainer comment listing TEST example IDs and alleged data problems.
Those IDs and allegations are quarantined: not copied into this artifact, not
looked up, not used for hypothesis generation, filtering, evaluation, or method
selection. The research question and resource search preceded their exposure.
No TEST dataset file, media, feature, score, or model evaluation was opened.
Do not describe this turn as having encountered no TEST-related information.
For later issue reads, select the known maintainer comment or filter author
association before displaying bodies. No issue, email or other message was sent.

## Action consequence

The proposed existing-human-label attribution route cannot currently support a
valid experiment. A new character encoder, pseudo-labeler, fingerspelling stream,
special crop policy or externally trained teacher would not resolve that
evidential gap and risks both prior-art and internal-closure collisions. Do not
launch one as a substitute. No candidate is promoted and no scientific family
is declared impossible. Continue the broader search in a different question;
retain the unchanged full GO criteria. No active worker or global blocker.
