# Discourse-context primary-source scan

## Material Passport and search scope

Origin: academic-research-suite / deep-research / three-way-scan, investigation.
Date and last searched: 2026-09-15. Status: ANALYZED. AI-assisted; no external
review, human-read attestation, full-paper reproduction or systematic-review claim.
No populated `literature_corpus[]` was provided for this particular question.

Search engine queries, in execution order:

1. `sign language retrieval discourse context cross sentence context coreference`
2. `PHOENIX 2014 T sentence segmentation context preceding sentences sign translation`
3. `document level sign language translation context previous sentences`
4. `"sign language retrieval" "context" discourse`
5. `"Lost in Translation, Found in Context" CVPR 2025`
6. `"Is context all you need" ICCV 2023 Sincan`
7. `"Neural Sign Language Translation" Camgoz 2018 CVPR 386`

English discovery, no year exclusion; primary author/publisher/dataset sources
only for evidence. Five papers retained for the narrow context question, plus
official PH documentation for its input contract. Camgoz et al. (2018) was also
located as the originating dataset paper; no numerical model result from it is
used. Reviews, mirrors, Wikipedia, search snippets of secondary summaries and
social media were excluded as evidence. SEDS was only a search hit, not accessed
as a resource. C²RL was recognized as the existing RPCA collision, not a new lead.
Title/author deduplication joins arXiv and proceedings versions. Programmatic S2,
OpenAlex and Crossref verification was not invoked; no API-match claim or invented
hit totals. Search-engine result dates are not treated as publication dates.

Four of five papers have publisher-confirmed conference/workshop publication;
DiscoSign is treated here as a preprint with author-reported EMNLP acceptance.
Coverage is heavily computational: 4/5 papers (80%) present computational models,
triggering a distributional-skew advisory. Response: retain the human baseline as
a counterweight; do not generalize model gains to linguistic necessity. There is
no claim that this small search exhausts sign-linguistic scholarship.

## WHY / HOW / WHAT and evidence limits

| Citation key and primary source | WHY | HOW | WHAT is usable here; verification scope |
|---|---|---|---|
| `tanzer2024`: Tanzer, G., Shengelia, M., Harrenstien, K., & Uthus, D. (2024). [Reconsidering Sentence-Level Sign Language Translation](https://aclanthology.org/2024.emnlp-main.360/). EMNLP, 6262–6287. | Check whether isolated clips preserve information needed by fluent signers. | Human translation under successively added context; 102 sampled How2Sign instances, two Deaf signer coauthors. | Authors report context-dependent key details in 33.3% of their sample. Read publisher metadata/abstract and original PDF §§2.2, 4.1–4.2 through browser, not full appendix examples. Initial sentences excluded; annotators knew the purpose; previous gold captions in later conditions are extra information. No PH retrieval prevalence or irreducibility estimate follows. |
| `sincan2023`: Sincan, O. M., Camgoz, N. C., & Bowden, R. (2023). [Is Context all you Need? Scaling Neural Sign Language Translation to Large Domains of Discourse](https://openaccess.thecvf.com/content/ICCV2023W/ACVR/html/Sincan_Is_Context_all_you_Need_Scaling_Neural_Sign_Language_Translation_ICCVW_2023_paper.html). ICCV Workshops, 1955–1965 (CVF pagination). | Disambiguate weak video evidence using prior context. | Video, spotting and preceding-sequence context encoders feeding a translation decoder. | Publisher abstract reports translation improvements on BOBSL/SRF. Establishes existing context-encoder mechanism, not independent-query retrieval efficacy. Abstract-level mechanism screen, no ablation reproduction. Surrey metadata uses different pagination; retain explicitly labeled CVF pages. |
| `jang2025`: Jang, Y., Raajesh, H., Momeni, L., Varol, G., & Zisserman, A. (2025). [Lost in Translation, Found in Context: Sign Language Translation with Contextual Cues](https://openaccess.thecvf.com/content/CVPR2025/html/Jang_Lost_in_Translation_Found_in_Context_Sign_Language_Translation_with_CVPR_2025_paper.html). CVPR, 8742–8752. | Supply information beyond signing features. | An LLM receives video features, automatically extracted background descriptions, previous translations and pseudo-glosses. | Publisher abstract reports cue ablations and BOBSL/How2Sign translation results. Generic previous-text/background conditioning is already explored. Abstract/introduction scope; official repository README also viewed, not its training implementation. |
| `yin2021`: Yin, K., DeHaan, K., & Alikhani, M. (2021). [Signed Coreference Resolution](https://aclanthology.org/2021.emnlp-main.405/). EMNLP. | Model spatial reference tracking in signed language. | Annotated German Sign Language coreference corpus, linguistic heuristics and unsupervised models. | Publisher abstract confirms an explicit task and annotation approach. Does not establish coreference labels in PH or identify our persistent errors. Abstract-level scope, no corpus acquisition. |
| `baltatzis2026`: Baltatzis, V., Inan, M., Gillis, C., Kushalnagar, R., Quandt, L., Findlater, L., & Lea, C. (2026). [DiscoSign: Discourse-Aware Text to Sign Language Gloss Translation](https://arxiv.org/abs/2609.02796v1). arXiv preprint, September 2. | Preserve spatial reference, question–answer-clause functions and concept/gloss consistency across discourse. | Modular LLM text-to-gloss translation with discourse-specific evaluation. | Original arXiv metadata/abstract inspected; acceptance only author-reported there. A fresh collision warning for generic discourse/coreference novelty claims, not a video-retrieval result. No full-paper efficacy verification. |

Official resource `rwthph`: RWTH Aachen. [RWTH-PHOENIX-Weather 2014 T: Parallel
Corpus of Sign Language Video, Gloss and Translation](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/).
Its Detailed Description states 386 forecast editions and interpreter-box-only
210×260 frames. This describes the release, not an independently verified mapping
from each local filename prefix to an edition. No archive or evaluation data was
downloaded.

## Source verification assessment

Existence and claimed titles/authors were checked on the primary URLs above.
Tanzer's DOI is publisher-listed as 10.18653/v1/2024.emnlp-main.360; DOI resolution
was not separately executed. No `S2_VERIFIED` or complete bibliographic audit label.

Provisional design/fitness: Tanzer III (within-instance context comparison), B for
its sampled human-task claim; Sincan/Jang III (reported model comparisons), C for
this abstract-limited mechanism screen; Yin VI (descriptive/task contribution), C
at inspected scope; DiscoSign III provisional, C/preprint for collision screening;
RWTH VI descriptive documentation, A for the published resource description only.
These are not grades for replicated SLRet effectiveness. COI: Tanzer's annotators
are coauthors and unblinded, explicitly disclosed by the paper; no full funding/COI
audit for the others. Publisher identity was checked (ACL/CVF); subscription
predatory-publication lists, author investigations and full retraction checks were
not performed. No observed source-integrity issue warrants fabrication claims.

No source was saved as a local original: `source_acquired=false`,
`source_verified_against_original=false`, `description_last_audit="none"` under
the skill's disk-backed passport semantics. This does not deny the explicitly
listed primary browser inspection; it prevents overstating local acquisition.
No private corpus was uploaded. A guessed `data/` directory was absent during
local filename discovery; that failed lookup is not evidence about corpus content.
The first direct CVF-2018 page open failed; subsequent primary search resolved it.
