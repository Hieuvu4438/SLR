## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (failure traceback and terminal process checks)
- Version Label: AS-C40-repair-v1

Four runs completed:42ON/OFF,1337ON/OFF. The queue then stopped during2026ON
after epoch12evaluation/169updates: `BrokenPipeError: [Errno32] Broken pipe`
at the progress print in augmentation_training_probe.evaluate. No optimizer,
nonfinite-loss or GPU-memory failure is recorded. Tool session9831expired;
process inspection found no queue/worker alive.2026OFF was never launched.
No restart was based solely on an expired observation handle.

Original2026ON_run.json has statusfailed and its entire folder, epoch0matrix,
partial epoch trace and console output history are preserved. No resumable
checkpoint exists under the registered no-checkpoint-write policy. Therefore
rerun that ONE interrupted condition from initialization as2026ON-attempt2,
then launch the still-unstarted2026OFF. Existing four complete runs are retained.
No selection of a favorable endpoint; same seed/data/recipe/schedule/horizon.

Worker change is limited to optional attempt argument, output-name suffix and
attempt metadata. No numerical training code changed. Both2026completed-policy
runs use this same updated worker source. Previous pairs both use original source;
validator compares source identity within each matched pair. Explicit2026ON
attempt2mapping is recorded in validator, not discovered by maximum performance.

Read-only source reconstruction verified that removing exactly the three
additions (attemptparser line, suffixguard block, attemptmetadata line) restores
the original worker SHA256
c6746557d34bed057db8faa6af81ff3b5f46d826021a411f407dc547b92a1201,
matching all four completed original runs ANDthe failed2026ONrecord. This
supports the output-only change claim; it is not a new training recipe.

New durable_augmentation_run.py routes stdout/stderr to exclusively-created
per-attempt console files, so tool observation lifetime cannot break training
progress prints. This alters only output transport. Protocol AS-C40_protocol.md
is unchanged. No original artifact is overwritten. All retries disclosed;
the broader autonomous research instruction authorizes this infrastructure
repair, while ARS requires preservation and honest failure reporting.

Commands: original900s timeout/PYTHONPATH, module durable_augmentation_run,
first --seed2026 --condition on --attempt2, then --seed2026 --condition off.
Monitor durable console, runJSON and process30–60s. Final validation requires
all six completed condition endpoints plus preservation of this failed attempt.

At retryepoch12, the rerun exactly reproduced the failed attempt's initial
state digest, all169batch traces (including augmentedinputs),13evaluation
records and181training/evaluation logs excludingtime/memory. Thus the output
repair preserves observed numerical learning through the entire failed prefix.
Fullretry endpoint still pending at this check. Fullprobe65tests passed3.04s.
