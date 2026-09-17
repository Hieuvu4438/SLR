# AS-C23 — omitted-token restoration versus length-matched repetition

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C23-v1

AS-C22 finds230/7096 TRAIN and10/519 DEV inputs omit content BPE tokens; no
new exact collisions. Three persistent T2V and two V2T errors involve affected
positive text. Test only the existence of useful omitted-token evidence in the
strong R0; no novel longer-context method claim or new labels/translation.

Frozen seed42 R0 video/text encoders, canonical519 gallery, unchanged positives.
First re-encode all clean32-token captions with historical batch128 and require
exact cached token equality; canonical scores <=2e-5 channel difference and exact
directional rank parity. Score matrix columns for unaffected inputs remain
BITWISE canonical. The rule is input length>30, never an error-label filter.

For each affected caption, compare:

1. Full original content tokens, preserving original order, SOT/EOT, all available
   within model77-position capacity. No new position interpolation or parameters.
2. Same total sequence length, but fill each omitted ORIGINAL position with the
   nearest retained token (lower retained index on equal distance). Preserve all
   originally retained tokens at their original positions. This controls length,
   position slots, valid-token count and encoder/scorer dimensions without giving
   omitted lexical identities. It is synthetic and not a semantic equivalence.

Both streams encode the full519 inputs in batches128 with77slots, but only
affected columns are rescored. For these columns, discard all hidden PAD slots
after EOT in BOTH conditions before scoring; original affected inputs occupied
all32slots and thus had no PAD. Full and control use exactly n+2 valid slots.
This prevents adding45 inner-softmax padding competitors. Unaffected columns
come unchanged from canonical scores, so their padding policy stays untouched.
No global max_words patch, test, training, dev selection, or weight sweep.

Report full R1/R5/R10, persistent mean ranks, affected-query ranks and all changed
V2T queries. Lead requires full >=+.5pp mean R1 over BOTH canonical and repetition,
no direction loss>.25pp, no R5/R10 loss>.5pp, and both persistent mean ranks improve.
Otherwise reject this exact frozen restoration. No three-seed pilot or GO from
one diagnostic. Context distribution shift and synthetic repetition limit causal
interpretation; changed token evidence may alter subsequent contextualization.

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.text_restoration_probe`
Cwd `/home/haipd/SLR`; timeout300s; monitor process plus`AS-C23-RESTORE_run.json`.
Outputs small matrices/per-query metrics only. Any failure preserved separately.
