# Pooling-prior transfer screen — Cycle13

2026-09-22. Academic-research-suite, inline source verification and adversarial
review. AI-assisted; no independent reviewer. No model execution, new annotation
or empirical SLRet result. This screen found no new admissible mechanism.

## Question and primary source

Does new evidence about late-interaction gradient concentration justify a
different intervention in the incumbent SEDS scorer?

Suresh et al., [*Spike Hijacking in Late-Interaction Retrieval*, arXiv:2604.05253v1](https://arxiv.org/html/2604.05253v1),
April2026: read abstract, sections2.2–2.4 and4.1–4.2. The synthetic soft-pooling
model distributes gradients more evenly but retrieves less accurately. The
real-document comparison swaps inference operators and injects distractors;
it is not SLRet. Results remain [A], not locally reproduced.

Provisional computational LevelIII, GradeC for transfer: fitnessC, reviewD
(unconfirmed), methodologyC, sampleC, currencyA, conflictsC (incomplete audit).
Author version names an ECIR workshop and Adobe affiliation. No independent
peer-review/funding/retraction checks, S2/DOI verification or empirical
corroboration. No predatory-publication accusation. Official arXiv confirms
existence; this is not a complete bibliography-verification gate.

### Mathematical caution, not an experimental refutation

The displayed section2.4 formula is

`F(g) = 1 - 2 * sum_j sum_(m<=j) g_(m) / (M * sum_j g_j)`.

Direct substitution with all `g_j=c>0` gives `F=-1/M`, not zero. Algebraically
it equals the pairwise absolute-difference Gini minus `1/M`. The same expression
appears in the [PDF text extraction, page3](https://arxiv.org/pdf/2604.05253).
PDF screenshot retrieval failed; no visual-rendering verification or author-code
inspection is claimed. This finding concerns the displayed equation only.

For equal M the offset cannot reverse comparisons; for varying M it varies.
It does not establish an implementation error, invalidate reported ranks, or
show that the paper's qualitative conclusions are wrong. Do not use the printed
expression uncritically as a new local diagnostic.

## Local transfer check

[SEDS source](../../../third_party/SEDS/modules/modeling.py:466) already computes
`sum(s * softmax(s / 0.07))`, then masks/averages outer tokens and scales logits.
The RGB and fusion branches use the same inner operator. It is not hard MaxSim.
Calling a hard-to-soft replacement a missing SEDS mechanism would therefore
misdescribe the control. Soft pooling can still concentrate; source identity
does not establish immunity or explain persistent errors.

The [historical AS-C24/C25 result](../../proposal7/evidence/autonomous_search/AS-C24_C25_result.md)
already tests six hard/mean replacements and three entropy interventions on
frozen CiCo. All nine reduce mean R1 in that regime. That is bounded historical
evidence, not a new SEDS experiment or a universal pooling impossibility claim.
No temperature, entropy, top-k or gradient-concentration rescue is admitted.

A second inspected idea, matching the loss to averaged channel logits, is also
not new: the [existing channel-loss audit](../../proposal7/evidence/autonomous_search/CICO_channel_objective_result.md)
already derives its agreement penalty and screens the obvious interventions.
No gradient census with no open decision consequence follows. Earlier checks
of anatomical coordinate recovery also encounter the recorded C07/C09/C21
designs; those are not newly discovered mechanisms.

## Decision and limits

Strongest counterargument: an exact operator match does not imply equal behavior
across temperatures, learned representations, masks or datasets. Agreed; this
screen rejects transfer of the proposed rationale, not every concentration
hypothesis. Natural PH rank harm and a distinct intervention remain unshown.
No new reviewer or semantic labels are required to establish a computational
mechanism, but missing evidence cannot be replaced by an adjacent paper's result.

Previous turn: PROGRESS through correction of an invented admission rule.
Current turn: **NO PROGRESS toward mechanism admission**; only a bounded source
qualification is added. No candidate changes status. Do not call this a new
bottleneck experiment, efficacy result, complete search or global blocker.

Searches included sign-retrieval/bottleneck/2026, late-interaction/gradient/token,
and current sign-retrieval/alignment queries. SAN, C2RL, SignMatch and SignSeek
were rediscoveries, not new leads. How2Sign's historical asset census was read
for scope only, not refreshed; no missing-file claim is made about its current
state. Historical TEST asset metadata appeared in that census and a broad text
search; no TEST examples, videos, score files or model evaluation were opened.
No paper, dataset or model was sent to an external model; no assets acquired.

Local source SHA256 values checked this turn:

- SEDS modeling.py: `62bd9aac7076e54a7ed15de947ca7437c5b7c78ec3b50216f9d733d1d09880c9`.
- AS-C24/C25 report: `ff3fd6e1e425e2694a82df7fecf388749335b5f3b642c175ff6f8809a2ed4267`.
- Channel-objective report: `187d3d5ff24905fc01b13a47e202195fb88de10fd2a1b036622245d783dc3a17`.

Hashing confirms the inspected file versions, not replication of their results.
