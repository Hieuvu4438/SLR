# AS-C29 — fixed checkpoint embedding-table rollback diagnosis

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C29-v1

AS-C28 measures drift in the two text tables frozen by the upstream default.
Does restoring their exact AS-C19 initialization at AS-C20 update1000 materially
change full1375 internal-held retrieval? Four fixed cells: identity; positional
table reset; token table reset; both reset. No interpolation/weight selection.
All other learned tensors stay bitwise unchanged; reload originals between cells.
Exact identity score replay required. Both resets use the generic-loaded then
target-cast AS-C19 initialization, not a different precision source. No optimizer
updates, official dev/test evaluation or new checkpoint. Save small held scores.

This is a POST-TRAIN intervention, NOT training with frozen tables. Learned
downstream layers co-adapted to the trained tables; a failed rollback cannot
reject freezing during training, and a successful rollback cannot isolate the
cause of original poor generalization. Nor can any gain over this weak model
satisfy the actual method GO gates. No generic freezing/rollback method proposed.

Practical diagnostic lead: >=.5pp held mean R1 gain, neither direction loses>.25pp,
R5/R10 losses<=.5pp. Report all three nonidentity cells; no significance or
winner selection. AS-C20 >=50% held-in-each-direction learning gate remains
unchanged and independent. The purpose is to bound direct current-table effects,
not to tune a residual generator to the held labels.

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.embedding_rollback_probe`
Cwd `/home/haipd/SLR`; timeout300s; process plus`AS-C29-ROLLBACK_run.json`.
Read TRAIN-only features and existing final checkpoint. Preserve failures; never
overwrite completed/failed runs or change the original checkpoint.
