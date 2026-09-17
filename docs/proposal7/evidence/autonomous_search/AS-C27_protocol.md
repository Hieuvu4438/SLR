# AS-C27 — common-model/common-gallery source-held versus official dev

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (before execution)
- Version Label: AS-C27-v1

AS-C26 finds similar simple lexical support but0/1375internal-held and402/519dev
rows share inferred source prefixes with the SAME5721fit rows. Historical R0
versus AS-C20 scores confound model, training regime and gallery size. Evaluate
the FIXED AS-C20 update1000 model on official DEV once and compare to519-row
internal-held galleries from the existing1375held score matrix. No training,
dev selection or new initialization. This is not a method gain versus R0.

Primary held subset: lowest519 SHA256("0:"+pair_id), sorted back to original
held order. Four additional fixed sensitivity subsets use salts1..4. Subsets
overlap and are NOT five independent models or replicates. Report all values
and their range; primary remains salt0 regardless of outcome. Restrict both
rows/columns to identical subset IDs so official singleton positives remain.
This diagnostic does NOT replace full1375held AS-C20 adequacy or official dev.

Load only generic architecture metadata plus saved clean update1000 weights;
verify checkpoint/partition/manifest hashes. Reuse AS-C20 float32 encoder and
scorer implementation, batch128/block64. DEV uses519existing features/captions;
all same-model held scores already saved. Record all R1/R5/R10 and per-query
ranks. Describe dev seen-prefix versus unseen-prefix R1 within the fixed full
gallery, explicitly using per-query T2V ranks for the subgroup. No p-values,
causal source advantage, semantic equivalence, information ceiling, or GO.

If both equal-gallery populations remain weak, do not blame the source partition
alone or lower the learning gate. If a gap appears, it is a descriptive common-
model difference with remaining content/domain confounding, requiring an actual
intervention for causal nuisance claims. Either outcome helps prioritize a
future adequate calibration protocol without turning generic training into a
novel method. No source correction, teacher or new targets authorized.

Command: `PYTHONPATH=shared:. /home/haipd/miniconda3/bin/python -m methods.information_probe.common_model_support_probe`
Cwd `/home/haipd/SLR`; timeout300s; process plus`AS-C27-COMMONMODEL_run.json`.
Save one small dev score matrix and metrics. No new model checkpoint or large
feature cache. Preserve failures separately; no silent retry.
