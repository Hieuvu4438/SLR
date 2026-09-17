# AS-C44 primary-source and annotation-provenance check

## Material Passport and scope

academic-research-suite / deep-research, targeted fact-check; 2026-09-15.
ANALYZED, AI-assisted. Public browser sources, not a systematic review, exhaustive
novelty search, local material acquisition, human-read attestation or reproduced
paper performance. No dataset archives, weights or third-party features downloaded.
Author papers/pages are primary but their shared authorship is not independent
replication. Bibliometric/predatory-database and comprehensive retraction checks
were not run; no such certification is claimed.

## Verified sources and bounded claims

1. Koller, Camgöz, Ney and Bowden, *Weakly Supervised Learning with Multi-Stream
   CNN-LSTM-HMMs to Discover Sequential Parallelism in Sign Language Videos*.
   [Author PDF](https://www-i6.informatik.rwth-aachen.de/publications/download/1099/Koller-PAMI-2019.pdf),
   [institutional record](https://publications.rwth-aachen.de/record/761009/),
   DOI 10.1109/TPAMI.2019.2911077. Read scope: abstract/introduction, stream
   synchronization, lexicon construction and experimental data sections.
   Primary peer-reviewed computational study, prior-art evidence, not our efficacy
   evidence. Sign/mouth/hand synchronization is explicit prior art. Mouth and
   hand training lexicons are weak mappings, not manual frame-level supervision.
   A separate handshape evaluation set is described as 3,361 PH2014 DEV images.
   Exact release-to-local split compatibility was not established.

2. Camgöz et al., *Multi-channel Transformers for Multi-articulatory Sign
   Language Translation*, [original record](https://arxiv.org/abs/2009.00299).
   Read scope: metadata and abstract only. Author-reported inter-/intra-articulator
   context gives a specific collision warning for generic multi-channel attention;
   no full-method equivalence, performance reproduction or exhaustive novelty
   verdict is inferred from the abstract.

3. *Spatial-Temporal Multi-Cue Network for Continuous Sign Language Recognition*,
   [original record](https://arxiv.org/abs/2002.03187).
   Read scope: metadata and abstract only. Primary multi-cue modeling precedent,
   not evidence that the present retrieval bottleneck has been solved.

4. [Official PH2014 handshape resource page](https://www-i6.informatik.rwth-aachen.de/~koller/1miohands-data/),
   read full text. The page reports **3,359** manually labeled PH2014 development
   images and 45 encountered handshape classes. The 3,359 versus 3,361 discrepancy
   with source 1 is unresolved; neither is a locally verified archive count.
   The resource's handshape task calls these test images, while its provenance
   names PH2014 DEV. Neither label establishes the split in our PH2014T-derived
   manifests. Single-frame handshape classes are not joint hand–face semantic
   contrast annotations. This is authoritative release documentation, not an
   independent evaluation or blanket permission to access local TEST examples.

5. Koller, Ney and Bowden, *Read My Lips: Continuous Signer Independent Weakly
   Supervised Viseme Recognition*, ECCV 2014.
   [Author PDF](https://www-i6.informatik.rwth-aachen.de/publications/download/933/Koller-ECCV%202014-2014.pdf).
   Read scope: title/abstract, introduction and §§2–4. Section 4 reports five
   annotated sentences per each of seven signers, totaling 3,687 frames, labeled
   three times by one knowledgeable non-native signer. This establishes that
   manual mouthing evaluation data exists, but not agreement among independent
   annotators, joint handshape labels or alignment to our current clips. Its
   corpus description is not interchangeable with the current PH2014T split.

6. [Author-linked Re-Sign repository README](https://github.com/huerlima/Re-Sign-Re-Aligned-End-to-End-Sequence-Modelling-with-Deep-Recurrent-CNN-HMMs),
   read README only via [author homepage](https://www-i6.informatik.rwth-aachen.de/~koller/).
   It explicitly links the 2017 and 2019 methods and describes frame-label
   generation, model posterior extraction and iterative alignment. This supports
   distinguishing generated alignments from manual annotation. It does **not**
   authenticate the contents or lineage of a particular downloadable archive.

7. [Official release directory](https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/),
   read listing only. It exposes named train_val and multi-stream alignment
   archives, a PH2014 DEV handshape archive and a mouthing archive. Names and
   sizes are discovery evidence only; members and split boundaries remain
   uninspected. No mixed archive was opened to inspect label rows.

8. [Official PH2014T page](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
   and [PH2014 page](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/),
   read descriptions and content lists. These distinguish releases and document
   interpreter-box input. The PH2014T README link returned HTTP 404; no README
   content or mapping was inferred. Published aggregate test tables appeared in
   browser text but no local TEST samples/scores/labels were accessed or used for
   selection.

## Search and decision record

Queries included: `sign language recognition simultaneous articulators non manual
temporal synchrony multi cue network`; `sign language partial information
decomposition synergy manual non manual`; `PHOENIX 2014 non manual mouth hand
annotations multiple streams synchronisation`; `Koller 2019 PAMI multi stream
synchronisation mouth hand shape sign language embedded`; exact handshape archive
name; exact train_val archive name; and multi-stream archive name plus README.
Search results were screened for primary sources. Secondary PID summaries,
ResearchGate copies and unrelated downstream papers were not claim evidence.
This search neither proves novelty absence nor proves no suitable annotation
resource exists elsewhere.

Immediate consequence: do not fabricate hand–face ground truth from spatial bins,
caption lexicons or model alignments. Preserve the possible manual-resource lead,
but require provenance, exact TRAIN/DEV matching and joint-contrast validity before
use. Human assistance was asked for optionally; no response is recorded, and the
broader autonomous search is not blocked. No candidate is promoted by this check.
