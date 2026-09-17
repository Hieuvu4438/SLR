# AS-C10 descriptive nuisance screen

Date2026-09-15; Q12/Q13, metadata/nuisance layer after optimization diagnostics.
Question: are persistent strongest confusers disproportionately same-source or
same-signer, even compared with a crude content-similarity control?
Use existing native-annotation forensics, official519 dev pairs and three saved
baseline matrices. No training, test, selector, semantic relabeling or correction.
For each query record its strongest incorrect candidate. Compare same-group rate
against uniform gallery availability and20 other candidates with nearest
seed42 pooled-text cosine to that confuser's cosine. Exclude self and confuser
from control neighbors. Report covariate gap, subgroup counts and descriptive
excess only; no causal significance inference from this matching heuristic.
No new method justified unless a later valid intervention supports nuisance
reliance. Generic domain invariance or prior-column correction is not novelty.
