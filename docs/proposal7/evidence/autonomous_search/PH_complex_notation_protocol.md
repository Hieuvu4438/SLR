# PH complex TRAIN notation: resource-content gate

2026-09-16. Academic-research-suite / deep-research, source verification inline.
Previous POST_C44 audit established exact TRAIN membership but expressly did not
interpret complex notation. This pass inspects its actual notation vocabulary,
not the excluded ECCV mouthing archive or an expansion of AS-C45 gloss pairs.

Read only the official standard and complex TRAIN CSVs, TRAIN manifest, release
README and evaluation shell source (do not execute its evaluation commands).
Read official 2012 annotation conventions for the semantics of original tags.
Do not assume that conventions survive unchanged into this transformed release.

Count exact tokens and row occurrence for documented literal prefixes lh-,bh-,
neg-,negalp-,negalpha-,poss-,loc-,cl-; literal postfix fields mb:,mk:,lh:,name:,
time:,loc:,obj:,in:; and every actual __...__ special token. Count variant #,
parentheses, -PLUSPLUS suffixes, and compound '+' as descriptive syntax only.
Report all predeclared counts, including zeros; no threshold selection or model
correlation. Preserve raw labels; do not create per-example cue labels, crops,
new positives or held-out splits. Validate complete TRAIN identity join and hashes.

Decision: only retained, documented cue-specific annotations could support a
later diagnostic design. A convention's existence or generic EMOTION/LEFTHAND
marker alone cannot identify exact visemes, spatial anatomy, simultaneous timing
or retrieval relevance. No model/feature/DEV/TEST access, no auxiliary training,
caption restoration, negative mining, generic stream or RCLI rescue. If the needed
markings are absent, stop this resource route without downloading other archives.
