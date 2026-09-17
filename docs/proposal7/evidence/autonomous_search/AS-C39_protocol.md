## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (registered before execution)
- Version Label: AS-C39-v1

## Historical first-epoch training replay gate

Before causal augmentation-on/off training, reproduce the original seed42
baseline first13updates exactly. Original selected best is epoch0step13.
Execute the unmodified `train` function recovered from git commit
39449e18def6b154944ceeaed39dfd5c570882a3, historical pathelsc/train.py.
Do not use today's initialization evaluation, absent in the historical function.
Do not shorten epochs in config: retain200epochs/2600total scheduler steps,
260warmup, originalbatch512drop-last, num_workers8, AMPbf16,AdamW,seed42,
word-swap augmentation, original release initialization and original manifest.
No baseline recipe adjustment or claimed upstream-PH recipe equivalence.

Compile ONLY that archived function into current trainer globals. Require AST
identity of baseline-active helper functions and mapped dataset/tokenizer/view,
retriever/adapter/loss/evaluation/runtime/RNG helpers. The factory builder differs
only in docstring and movement of its retriever import to module scope; verify
normalized AST. CiCoBridge paired_score differs but baseline uses score, not
paired_score; require identity of active encode_video/encode_text/score methods.
Record archived blob and current source hashes. No auxiliary ELSC objective,
teacher, new positives, SEDS asset or test access; method MUSTbe baseline and all
auxiliary weights zero. Existing config contains unused test paths, never loaded.

Wrap evaluate_model to save the one epoch0DEVscore matrix, then return its
unchanged result. Replace only checkpoint-save callback: at first last.pt save,
require epoch0step13 and compare actual model, optimizer, scheduler, AMPscaler,
RNG and sampler-generator state against archived best_dev.pt. Compare actual
trainable names. This callback stores a compact comparison report and ends via
an intentional sentinel before any later epoch or checkpoint write. It does
not change any forward/backward/update/schedule operation. No checkpoint save.

Require all model tensors exact, optimizer tensors/scalars and scheduler/scaler
exact, RNG/sampler states exact, all13training records exact except elapsed time
and peakGPU memory, and epoch0full score matrix exact. Preserve detailed failing
paths/deltas if anything differs; no silent tolerance or automatic ablation.
New/removed log fields are mismatches, not silently ignored. Selection/provenance
metadata outside the numerical replay scope are not declared bit-identical.
No training effect/augmentation benefit can be inferred from this replay alone.

## Execution

    PYTHONPATH=shared:methods/elsc:. timeout 300 /home/haipd/miniconda3/bin/python -m methods.information_probe.historical_training_replay

Expected tens of seconds and~43GBGPU, hard300s limit. Check current GPU capacity
first. Output AS-C39-TRAIN-REPLAY_run.json, separate AS-C39training output folder,
one epoch0DEVscore matrix, no overwrites. Monitor process/output30–60s. Unit tests
before launch cover archived function identity and recursive exact comparisons.
Record any failures; no experiment retry hidden. Full11/11fallacy scan follows.
If exact replay fails, investigate attribution before testing augmentation;
do not turn recipe repair into a novel method or optimizer rescue sweep.
