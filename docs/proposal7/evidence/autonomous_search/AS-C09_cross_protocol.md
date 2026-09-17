# AS-C09 endpoint crossing protocol

Date2026-09-15. Historical trajectory scan found non-monotonic, seed-dependent
selection (epochs0,-1,146); per-query order validated against all saved selected
ranks and pinned sequential evaluator. Inspect best/last weights only, train/dev
scope. For each seed encode full dev once with each endpoint, then cross video
and text states into four fixed conditions. No fitting/selector/new training.
Require bit-exact best/best score replay. Use one fixed positive scalar logit
scale within a seed; rank invariance means scale drift is not a ranking mechanism.
Report full-gallery metrics, source provenance, token cosine drift. Crossing can
break coordinate compatibility; do not conclude the tower with larger cosine
change is harmful or that hybrids are a method. No inference of linguistic
information loss without controlled probing. No Proposal8 or RPCA reopening.
