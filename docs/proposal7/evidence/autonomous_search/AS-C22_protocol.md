# AS-C22 — deployed text omission and error-support inventory

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C22-v1

Question: how often does the actual uniform30-content-token selection (plus
SOT/EOT) remove BPE tokens, merge otherwise distinct complete token sequences,
and coincide with historical persistent dev errors? No assumption that an
omitted token is a sign-relevant contrast. No model/metric or positive changes.

Use all7,096 TRAIN and519 DEV caption_model strings, the shared CiCo tokenizer,
and existing three-seed dev score matrices. No test, new translation, native
caption substitution, label fitting, auxiliary model, or feature cache needed.
Validate a transparent index reconstruction against encode_cico_text for EVERY
row. Record complete content length, selected/omitted indexes and token strings,
new exact collisions relative to complete token IDs, length summaries and
counts beyond the generic77-token context (75content). Record per-query T2V
and V2T historical ranks and fixed strongest-confuser identities.

Strata: affected versus unaffected queries, historical persistent errors,
persistent top10 errors, and V2T errors with affected positive/confuser text.
Use one optimistic T2V rank/query for subgroup counts, explicitly not its official
tie-expanded aggregate. Exact-sequence collisions are algorithmic, not semantic
equivalence or new relevance annotations. V2T may also change via a different
affected candidate outside the strongest-confuser pair; do not claim the pair
inventory is an upper bound. No statistical significance or causal effect claim.

If at least3 persistent errors involve an affected query text(T2V) or positive/
fixed confuser text(V2T), a separately preregistered frozen diagnostic may examine
restoration with context/padding controls. This is only a small-screen feasibility
rule, NOT a method threshold. Otherwise defer this low-support mechanism. Never
use dev labels to choose a subset at inference; any intervention must follow an
input-only rule across the full gallery. Extending a limit alone is not novelty.

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.text_omission_inventory`
Cwd `/home/haipd/SLR`; timeout300s; process plus `AS-C22-OMISSIONS_run.json`.
Save only local JSON audit, no user-file replacement. Autonomous authority permits
this diagnostic; preserve any execution failure and disclose corrections.
