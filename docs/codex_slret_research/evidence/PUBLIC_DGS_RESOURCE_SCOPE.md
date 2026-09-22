# Public DGS Corpus: verified annotation lead, unresolved use scope

2026-09-22. academic-research-suite, fact-check mode, inline source verification.
AI-assisted; no independent linguistic review. **Resource lead only**, not a
candidate promotion, PHOENIX relevance judgment or new evaluation benchmark.

## Sources and bounded verification

1. [Annotation Conventions v4.1, Konrad et al. (2022)](https://www.sign-lang.uni-hamburg.de/dgs-korpus/arbeitspapiere/DGS-Korpus_AP03-2018-01v04.1_en.pdf).
   Read title/version, introduction, translation, segmentation, lemmatisation,
   double glossing, double-token tags and initial glossing-convention sections
   in browser-extracted text. Later mouthing-section screenshots timed out;
   no claim of reading that section or the full document.
2. [Data Statement v3, Schulder et al. (2024)](https://www.sign-lang.uni-hamburg.de/dgs-korpus/arbeitspapiere/DGS-Korpus_AP06-2020-01v03_en.pdf).
   Read title/version, header, executive summary, curation and §3.2.2–3.2.3.
   This statement's dataset header says version3; it must not be conflated with
   the separately observed Release4 portal.
3. [Release4 transcript1206131](https://www.sign-lang.uni-hamburg.de/meinedgs_r4/html/1206131_de.html).
   Inspectable transcript table and download-category headings; no ELAN payload,
   video, pose file, archive, or full release downloaded or parsed.
4. [Release4 English license](https://www.sign-lang.uni-hamburg.de/meinedgs_r4/ling/license_en.html)
   and [German license](https://www.sign-lang.uni-hamburg.de/meinedgs_r4/ling/license_de.html),
   both read in full. Version-independent dataset DOI failed to open in this
   browser; do not claim DOI resolution or verification of the latest release.

Evidence grade: official descriptive project documentation (LevelVI; strong for
its own schema/release statements, not independently reproduced accuracy).
Project-authored web terms are direct evidence of posted conditions, not legal
clearance. Predatory-journal screening is N/A for these institutional records;
no peer-review claim. All sources share institutional provenance, so agreement
is not independent corroboration. API bibliographic verification was not run.

## Verified facts relevant to Q17

[V] The conventions distinguish parent sign types from conventional form–meaning
subtypes and provide linked tokens. Subtype differences can involve mouthings.
Manual token boundaries and type/subtype tiers are documented, with separate
hand tiers in ELAN. These are existing corpus annotations, not new labels.
[Conventions, lemmatisation/double-token sections](https://www.sign-lang.uni-hamburg.de/dgs-korpus/arbeitspapiere/DGS-Korpus_AP03-2018-01v04.1_en.pdf).

[V] Mouthing labels represent fully realised target words, not exact visible
articulation. Their intervals are anchored to sign boundaries and can cover
multiple signs. English translations are freer re-translations of German.
[Data statement §3.2.3](https://www.sign-lang.uni-hamburg.de/dgs-korpus/arbeitspapiere/DGS-Korpus_AP06-2020-01v03_en.pdf).

## Permission boundary

[V] Posted terms restrict download/use to linguistic research and direct readers
to contact the project for other research/applications; the English contact
sentence is broader than the German wording. Neither page expressly clears our
retrieval-model experiment. [English terms](https://www.sign-lang.uni-hamburg.de/meinedgs_r4/ling/license_en.html),
[German terms](https://www.sign-lang.uni-hamburg.de/meinedgs_r4/ling/license_de.html).

[U] Whether this project's intended computational diagnostic is covered or has
separate permission. Do not infer permission from public visibility or from
other papers using the corpus. Ask whether the user already has project-specific
permission; do not contact the maintainers or acquire corpus assets automatically.
This is a resource-specific gate, not a global research blocker or a request
for new human annotation.

## What this could change — inference and limitations

[I] There is now a concrete same-language source to investigate a **within-parent
type/subtype distinction**, rather than assuming that differing PHOENIX gloss
strings isolate a nonmanual cue. This does not establish matched visual pairs,
actual model failure, sufficient sample counts or an admissible method.

If use is cleared, the next bounded step is schema/sample feasibility: verify
original token IDs, parent links, intervals and recording/signer separation in
a small unchanged annotation export. Do not generate new captions or relevance
labels. No model fitting is authorized by this note. A later diagnostic would
need controls for context, signer and manual variation; parent membership is
not proof that two videos differ only in mouth movement. Do not treat target-word
labels as exact lip transcriptions or use their intervals as precise synchrony
ground truth. Causal attribution and relation-head novelty remain unestablished.

No PHOENIX join exists here. This corpus cannot adjudicate its confusers or
replace its locked retrieval gallery. Any future use must be disclosed as an
external diagnostic/resource, not hidden additional supervision. Existing
RCLI, generic face-stream, lexical-auxiliary and alignment closures still apply.

Search scope: local `docs/` and `research/` text for DGS Corpus/MY DGS variants
found no prior dedicated record; this is not proof of first discovery in all
project history. Bounded web queries covered official DGS annotation/download
and license pages. No exhaustive literature review or novelty claim.
