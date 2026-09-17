## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (registered before execution)
- Version Label: AS-C31-v1

## Question and locked scope

Are causally encoded common-prefix text tokens a broad adverse score component
among strong R0 persistent confusers? Code fact: early text tokens cannot access
later content. Later/EOT tokens can, so this is not automatically a model defect.
No attention mask, score rule, training, positive relation, or gallery changed.
TRAIN prefix inventory, DEV full519 gallery diagnostics, no test access.
R0 seed42 selected checkpoint and its fixed caches only; not the weak AS-C30
model. Persistent populations are unchanged3-seed intersections92T/87V.

## Definitions and falsifiable descriptive hypothesis

Deployed32-token input, content excludes SOT/EOT/padding. A common prefix is an
exact BPE prefix shared by at least two DIFFERENT complete deployed inputs.
Inventory TRAIN/dev prefix groups/rows/slots and actual token-vector equality.
Independently reencode every train/dev caption and require exact cache equality.
Verify actual32x32 causal mask equals triangular negative-infinity mask.
Numerical prefix-vector differences, if any, are measured rather than suppressed.

DEV fixed hardest non-diagonal confuser in EACH direction uses baseline scores;
retain but explicitly exclude identical deployed-text confusers from the burden
test. Compare common-prefix length>=3 against each query's entire other-text
gallery excluding identical deployed inputs. Descriptive counts only, no
caption length/content matching or semantic equivalence inference.

For each positive/confuser pair, decompose the UNCHANGED mixed score into text
token additive credits. With dot matrix a[f,l] and tau=.07:

    A_credit[l] = scale / N_valid_video * sum_valid_f a[f,l] softmax_l(a/tau)
    B_credit[l] = scale / N_valid_text * valid_text[l] * sum_f a[f,l] softmax_f(a/tau)
    mixed_credit[l] = (A_credit[l] + B_credit[l]) / 2

Inner padding/CLS remain exactly as deployed. Sum of all credits reproduces
the score. Group positive-minus-confuser credits into SOT, common content prefix,
remaining content, EOT and padding; use each caption's own EOT/length in V2T.
These are additive numerical terms, NOT causal token importance. A weights depend
on all text tokens and B normalization depends on caption length. Changing
attention/inputs changes other terms; deleting a credit is not an encoder test.

Broad-burden hypothesis: at least20% of the fixed persistent queries in EACH
direction have different text inputs, common content prefix>=3BPE, actual margin
<−1e-4, common-prefix credit margin<−1e-4, and all remaining credit margins sum
>1e-4. This is a deliberately explicit broad-signature threshold, not a GO gate.
Report all counts/directions and whole-gallery reference rates even if it fails.
Do not infer absence of all right-context limitations from this one signature.

## Integrity, interpretation and execution

Full-gallery channel replay tolerance2e-5 and all baseline ranks exact.
Paired channel-credit sum error<=5e-5; bucket margin sum error<=1e-4.
No tolerance change after execution. Unit tests cover prefixes, bucket partition,
both-channel conservation and inclusion of inner text padding. No overwritten
prior attempts. Fixed run JSON AS-C31-PREFIX_run.json retains provenance,
per-query records and summary. Timeout300s; process/output monitoring30–60s.

    PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.text_prefix_audit

Expected seconds to tens of seconds, under4GB GPU, read-only model/cache use.
No new relevance annotations or caption upload. If broad signature fails, do not
launch a bidirectional-text model merely from the mask's existence. If it passes,
require a controlled context intervention and targeted primary-source novelty
screen before candidate development. Common-prefix correction may collide with
closed nuisance/candidate-prior families; do not rename that mechanism.
