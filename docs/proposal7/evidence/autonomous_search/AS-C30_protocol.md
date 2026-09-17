## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (preregistered before execution)
- Version Label: AS-C30-v1

## Question and boundary

Does freezing exactly the two text tables identified in AS-C28 improve the
clean TRAIN-fold calibration enough to meet its ORIGINAL diagnostic adequacy?
AS-C29 inference rollback is not training-time freezing. This is one calibration
factor, not a new method, an upstream recipe reproduction, or a method GO pilot.

## Locked comparison

Control: rerun immutable AS-C20 source SHA256
e812d98277cd7f7d00c02600722ba53a5d2dafc767f3db44df6a26f572d7dc71.
Freeze: identical execution, except requires_grad=False for
clip.positional_embedding and clip.token_embedding.weight before training.
No other parameter is frozen. Frozen parameters remain in the same AdamW
groups; absent gradients cause AdamW to skip their updates/decay. Global
clipping consequently sees no gradients from those two tables: this is part of
the freeze intervention, not an independent optimizer change.

The harness parses the original source and changes exactly four string literals
for run JSON, artifact directory, experiment ID, and protocol path. Tests reverse
these substitutions and require AST identity. The original source is never
edited. Original initializer and dump are wrapped solely to set/audit freezing,
record provenance, and compare results. Transformed AST digest is recorded.

Both arms use generic CLIP initialization seed42, PH TRAIN5721 fit/1375 held,
original source-prefix partition and fixed1375 fit evaluation subset; original
FP32, batch128, augmentation,1000updates,lr schedule,AdamW,beta,epsilon,decay,
global clipping and no logit-scale clamp. Evaluate fixed steps0/250/500/1000;
final1000 checkpoint only. No dev/test access, PH-fitted checkpoint, or new data.
No time-based checkpoint selection, hyperparameter sweep, or early-stop rescue.

Control must exactly match AS-C20 all8 score files/arrays, all evaluation
records,40 training entries and every final state tensor before freeze may run.
Checkpoint container bytes differ intentionally because protocol metadata differs.
If mismatch, preserve and diagnose; do not silently relax tolerance or retry.
Freeze must leave both tables exactly equal to initialized FP32 values.

## Outcomes, decision, execution

Primary descriptive contrast: final held mean R1 freeze minus control, plus
both directional R1/R5/R10 and fit R1. No significance or cross-seed claim.
Original adequacy: BOTH fit R1>=80, BOTH held R1>=50, held mean improvement
from initialization>=5pp, and>=50 strict different-input held errors in EACH
direction. Weak-control gain does not replace this absolute gate.
Failure closes this isolated table-freeze calibration factor; it cannot show
an information ceiling. Other audited optimizer differences remain unresolved,
but freezing/optimizer corrections alone never count as novelty.

Commands, working directory /home/haipd/SLR:

    PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.freeze_training_comparison --arm control
    PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.freeze_training_comparison --arm freeze

Outputs: AS-C30-CONTROL_run.json and AS-C30-FREEZE_run.json here; corresponding
artifacts/proposal7/phase2/AS-C30-{CONTROL,FREEZE} directories. No overwrite.
Timeout3600seconds per arm, process/output monitoring every30–60seconds.
Expected about7–10minutes and<=20GB GPU per arm, about1.3GB disk total.
Report all failures, exit status where recoverable, fixed endpoints and all11
statistical fallacy checks. Prior-cycle results and original source stay intact.
