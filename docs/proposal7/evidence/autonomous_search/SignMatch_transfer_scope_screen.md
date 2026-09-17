# SignMatch transfer-scope screen

Date: 2026-09-16. Status: ANALYZED; adjacent prior, no admitted candidate.
AI-assisted targeted search and primary-source reading using
academic-research-suite. Not a systematic review, full-paper review, independent
replication, or source-code audit of SignMatch.

## Question and verified evidence

Can a newly located sign-retrieval mechanism justify an experiment within our
existing sentence text–video task, permitted inputs and closed-family boundary?

Wong, R., Jang, Y., Momeni, L., Varol, G., & Zisserman, A. (2026).
*SignMatch: Matching Dictionary Signs to Continuous Sign Language Video*.
The [Oxford publication record](https://robots.ox.ac.uk/~vgg/publications/2026/Wong26/)
identifies BMVC 2026; the inspected manuscript is
[arXiv v1](https://arxiv.org/html/2609.01886v1), not an independently verified
camera-ready implementation.

Evidence from the manuscript, abstract/introduction and §§3.2–4: the target is
visual sign-form matching, not sentence semantics. Stage 1 learns sign-class
prototypes from temporally annotated continuous signing. Stage 2 freezes that
space and learns a dictionary-video mapping. Training uses BSLCorpus sign
intervals/id-glosses and BSL SignBank; RGB initialization is SignRep Video-Hiera.
Larger-scale training adds pseudo-labelled corpora. These are substantive input
and supervision differences from our existing paired-sentence feature setup.
No reported benchmark number is used here to infer PH sentence-retrieval gain.

The [official project page](https://www.robots.ox.ac.uk/~vgg/research/signmatch/)
still labels code as forthcoming and links no model implementation. Its footer
source-code link concerns the webpage template. Its paper availability text is
stale relative to the independently accessible manuscript. This is a bounded
page observation, not proof that no code exists elsewhere.

## Transfer decisions — our inferences, not the authors' claims

| Possible transfer | Missing justification / collision | Decision now |
|---|---|---|
| Import the trained sign representation and dictionary alignment | Changes backbone, annotation and resource regime; does not isolate a new mechanism under our controls | Not a matched baseline or candidate experiment |
| Mine local pseudo-labels and use them to supervise our retrieval model | Needs a valid permitted supervisory path and an independent causal mechanism; straightforward teacher/support mining overlaps ELSC/SSSC closures | Do not relaunch those families |
| Add unlabeled prototypes to the existing sentence features | Removes the paper's sign-class/temporal supervision; no evidence yet that these prototypes retain its intended function or address our persistent errors | Untested idea, not an admitted candidate |

The last row is NOT a proof against all prototype models. Neither adding a
prototype layer nor changing its loss establishes novelty. A viable future
proposal would first need a measured permitted-data bottleneck, an intervention
outside the [closure registry](../../Negative_Results_Registry.md), relevant
prior-art checks and controls that distinguish its mechanism from capacity and
ordinary regularization. No new Q-number or three-candidate gate is claimed.

## Search record and exclusions

This turn used web discovery queries including:

```text
"sign language retrieval" 2026 method github
"sign language retrieval" 2025 fine grained representation github
"sign language retrieval" duration temporal position
"SignMatch" sign retrieval robots ox
"sign language" "Mul-Net"
"Compress and Forget" sign language
```

Some variants excluded previously read method names and secondary aggregators;
search engines still returned repeated or irrelevant hits. Results were checked
against the existing literature log. SignSeek, GTRN and SignRep were already
screened; their rediscovery is not new evidence.

The [WSLP pose-spotting paper](https://aclanthology.org/2025.wslp-main.10/)
was rechecked at abstract/metadata scope: binary query-sign presence is not the
target full-gallery sentence retrieval task. This confirms an existing exclusion,
not a new finding. *Compress and Forget*
([primary abstract](https://arxiv.org/abs/2608.18578)) concerns LLM context-memory
quantization, so was discarded as an irrelevant search hit. Secondary topic pages
were discovery aids only, not supporting sources.

## Limits and handoff

Read scope: SignMatch metadata/project page, abstract/introduction and selected
method/training sections; not the complete paper, supplement or implementation.
Published evaluation passages were encountered during browsing but were not used
for candidate selection or a numerical comparison. No benchmark TEST data was
accessed or evaluated. No datasets, annotations, model assets, SEDS assets or
checkpoints were downloaded; no training, GPU run or upstream edit occurred.

Outcome: one newly verified adjacent prior, no demonstrated implementation
weakness, empirical improvement, GO or global research barrier. Avoid another
unchanged code-availability lookup or prototype sweep merely to continue activity.
The next branch must provide new admissible evidence about an enabled mechanism;
it must not repeat the completed SAN/SEDS gates or count this scope screen as
empirical progress toward the numerical GO threshold.
