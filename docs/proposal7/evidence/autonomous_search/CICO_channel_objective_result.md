# CiCo channel-loss decomposition: verified identity, no harmful-mismatch claim

## Material Passport

2026-09-16; academic-research-suite, inline source analysis, mathematical
verification and targeted primary-literature check. ANALYZED, not a method pilot.
AI-assisted; no independent reviewer or human sign-language validation.

## Finding

For the inspected `.5` balanced configuration, CiCo's separate-channel loss
equals bidirectional cross-entropy of the averaged logits **plus a nonnegative
channel-agreement term**. This is a distinct axis from the earlier retrieval-
direction gradient audit, but a known ensemble-loss structure, not a new theorem
or an implementation bug. It does not establish that the term harms retrieval.

No mixed-loss, disagreement-regularization or gradient-surgery training campaign
is justified by this identity alone. This records a source-grounded interpretation
and a prior-art boundary for existing Q14/Q16, not a new numbered research branch.

## Source scope

SLRT revision `38a4f7b00da7a858d59b7fabe5093876a84db8e0`:

- [model forward](../../../../third_party/SLRT/CiCo/CLCL/modules/modeling.py:273)
  takes four cross-entropies in the inspected balanced `.5` configuration.
- [CrossEn](../../../../third_party/SLRT/CiCo/CLCL/modules/until_module.py:194)
  is negative mean diagonal row log-softmax, not a different contrastive kernel.
- [deployment bridge](../../../../shared/slr_common/upstream/cico_bridge.py:93)
  mixes the channel matrices before retrieval.
- [resolved seed42 configuration](../../../../runs/ph_base_b512_s42/resolved_config.yaml:51)
  specifies `dual_mix: 0.5`, `mix_design: balance`; this is not a claim for every
  configuration or every author's published experiment.

Source hashes: modeling `aeaf646b720f4cf051950dcbac68347196bed4b1c856d9aa833b063ed09e9e7f`;
CrossEn file `9fad15316a5fdc3133ffa2bd1e61b0bc94a9b4b5699af75662ccd1c1fd7538db`;
bridge `f1e524761f4d08f0cbce61d7a61c3b074ae2249c074ea12f7a1817587cb1a45e`;
config `06b24e4f6270d61c516e943b73aa76f9eacfe7a5e94d043431b8f3111fb1293d`.

Important: training channel A can use clean text while B uses augmented text.
The algebra applies to those two numeric matrices too, but their mean is then
**not the clean-text deployment score**. Do not claim the entire train/inference
difference is exactly this agreement penalty. The existing C40 augmentation
intervention remains a separate, failed fixed-horizon lead.

## Exact derivation

For one query, let logits be a,b, positive index y, m=(a+b)/2,
p=softmax(a), q=softmax(b), and r=softmax(m). Then

`r_j = sqrt(p_j*q_j) / sum_k sqrt(p_k*q_k)`.

Writing CE(a,y)=logsumexp(a)−a_y gives

`[CE(a,y)+CE(b,y)]/2 − CE(m,y)`
`= [logsumexp(a)+logsumexp(b)]/2 − logsumexp(m)`
`= −log(sum_j sqrt(p_j*q_j)) = J(a,b) >= 0`.

Nonnegativity follows from Cauchy–Schwarz; equality holds exactly when p=q.
J is independent of which y is designated positive. Equivalently,
`J = [KL(r||p)+KL(r||q)]/2`; it is not the arithmetic-mixture Jensen–Shannon
divergence. Row-constant logit offsets preserve a row's probabilities, but need
not preserve the column probabilities in bidirectional retrieval.

For square paired matrices A,B and M=(A+B)/2:

`L_sep = [CE(A)+CE(A.T)+CE(B)+CE(B.T)]/4`
`L_mix = [CE(M)+CE(M.T)]/2`
`L_sep = L_mix + [mean_row J(A,B)+mean_column J(A,B)]/2`.

This is an equality of objectives. It is **not** a claim that gradient descent on
L_sep separately minimizes each summand, that their gradients conflict, or that
removing J improves unseen ranks. In logit space, A→A+D, B→B−D leaves M unchanged
but can change J. CiCo's channels share token affinities/encoders: arbitrary D
may not be realizable by an encoder update. No realizability claim follows.

## Numerical source check

[Protocol](CICO_channel_objective_protocol.md) preceded execution.
[Implementation](../../../../methods/information_probe/channel_objective_identity.py)
extracts only the actual CrossEn class by AST, with no upstream module import.
[Retained JSON](CICO-CHANNEL-IDENTITY.json) is the second successful execution;
the first also exited0 and printed the same four fixture results. No independent
implementation of autograd is claimed.

| Fixed FP64 4x4 fixture | Separate loss | Mixed loss | Agreement gap |
|---|---:|---:|---:|
| Identical | .053490450 | .053490450 | numerical zero |
| Row offset | .124783255 | .078320081 | .046463175 |
| Opposite perturbations around4I | 1.430821949 | .053490450 | 1.377331499 |
| Opposite global offsets±1000 | .053490450 | .053490450 | numerical zero |

All loss-identity errors <=5.21e−17; derivative identity errors <=2.78e−17.
Actual CrossEn versus separately written logsumexp loss error <=1.49e−14,
within locked1e−10. Equality cases have tiny negative roundoff (~−5.2e−17),
not negative theoretical divergences; no clipping hid them. These are synthetic
logit fixtures, not paired videos, four model seeds or measured failure prevalence.
No GPU/checkpoint/dataset/DEV/TEST evaluation, optimizer update or score selection.
Existing probe regression suite:90passed in1.96s, exit0. The four fixture checks
run separately through the new module's main; they are not four extra collected
pytest tests. No independent full-model validation or efficacy test is claimed.

## Prior-art screen and counterevidence

Searches, 2026-09-16: official CiCo title; ensemble cross-entropy/average logits/
Bhattacharyya/ambiguity decomposition; exact title of the collusion paper.
Only primary official proceedings/journal pages and PDFs support this decision.

- [Wood et al., *A Unified Theory of Diversity in Ensemble Learning*, JMLR
  24(359), 2023](https://www.jmlr.org/papers/v24/23-0041.html): official metadata,
  PDF Proposition3 and Corollary9/logit-averaging discussion read. These establish
  prior ambiguity decompositions and normalized geometric-mean combination.
  Do not claim discovering the separate-versus-mixed loss decomposition.
- [Jeffares et al., *Joint Training of Deep Ensembles Fails Due to Learner
  Collusion*, NeurIPS2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2bde8fef08f7ebe42b584266cbcfc909-Abstract-Conference.html): official abstract and
  PDF introduction/background excerpt read. It studies individual versus joint
  training, including interpolation and cancellation that fails to generalize.
  This is counterevidence to automatic benefit from matching the combined loss,
  **not** a reproduced failure of CiCo's coupled channels. Full experiments and
  author code were not audited.
- [CiCo official record](https://openaccess.thecvf.com/content/CVPR2023/html/Bao_CiCo_Domain-Aware_Sign_Language_Retrieval_via_Cross-Lingual_Contrastive_Learning_CVPR_2023_paper.html)
  verifies the paper identity. Direct proceedings-PDF and attempted arXiv-HTML
  retrieval failed in this check. The source observations above are therefore
  not presented as a newly verified discrepancy against the complete paper.

These primary ML papers are suitable mechanism/theory evidence, provisionally
LevelIII computational studies with explicit theoretical results; no claim of
SLRet effect, comprehensive novelty search, full-paper reading, independent
replication, COI/retraction audit or DOI-resolution verification.

## Candidate boundary and next action

Three obvious formulations do not qualify as new methods:

1. Train CE of the mixed logits: established joint-ensemble objective and an
   ordinary objective change; no local harmful-penalty measurement.
2. Interpolate mixed and separate losses: algebraically adjusts J's coefficient;
   explicitly adjacent to the inspected interpolation prior, not a new mechanism.
3. Project or protect gradients to retain mixed-score ranks: overlaps the closed
   generic conflict-control/RPCA family. A different name or channel axis does
   not reopen it.

[C12](AS-C12_protocol.md) and C18 compared retrieval-direction gradients, not
separate-versus-mixed channel losses. Their positive cosines cannot close this
different empirical question. Conversely, its unmeasured status alone does not
justify training or a further diagnostic with no open decision consequence.

No train-gradient census or joint-loss efficacy campaign follows. Q14/Q16 remain
unresolved beyond their bounded evidence; the above three proposed formulations
are not promoted. Continue elsewhere in the existing map only with new source-
grounded evidence and an open decision consequence. No Q38, Proposal8 or GO.
