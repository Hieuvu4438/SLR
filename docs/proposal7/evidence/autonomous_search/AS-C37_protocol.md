## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: UNVERIFIED (registered before execution)
- Version Label: AS-C37-v1

## Question and locked boundaries

Do valid temporal slots repeat exactly within original PH TRAIN/DEV clips,
despite the absence of whole-input collisions in AS-C36? Census all7096TRAIN
and519DEV rows using the same canonical64-slot,alpha.9 loader. Exclude padding.
Check FP32fused inputs and nativeFP16cast separately; no encoder execution,
near-similarity threshold, caption/rank-conditioned selection, or parameter sweep.

Pin the completed AS-C36 run hash before launch. Recheck both manifests, source
code hashes, both feature-file hashes per row, row identity, view hash, valid
length, and both complete-input digests against AS-C36. An inconsistency fails
closed. No test split or external upload, source/label/checkpoint modifications.

For each precision, group valid row vectors by canonical signed-zero byte
equality with dtype/shape fixed. Require torch.equal for every repeated pair.
Report valid/unique/excess-slot counts, all unordered repeated pairs, adjacent
sampled-slot pairs, and group slot lists. One-slot rows have zero eligible pairs;
include them in row census, not an invented repetition percentage denominator.

For every clip with a repeat, reload both original dense streams and require
their fused selected rows exactly reproduce loader output. For every repeated
pair, classify mutually exclusively: same dense index; distinct indices with
equal FP32fused vectors AND both source streams equal; distinct indices with
equal FP32fused vectors but not both streams equal; or FP16-only merger of
unequal FP32vectors. Also report marginal aware/agnostic equality counts.
The dense source streams are float32 after the original loader conversion.
These checks locate repetition in current feature artifacts, not raw RGB.

Report TRAIN/DEV separately at both precisions, census denominators, source
verification counts, and per-row records. Any source-attributed repeat warrants
interpretation, NOT a defect verdict: static content and intentional extraction
overlap remain alternatives. Zero repeats rejects only this exact temporal
collapse explanation. No precision, sampler, deduplication, or alignment method
is justified by the inventory alone. No causal retrieval or information-ceiling
claim. After this second bounded acquisition check, switch layer if it yields
no attributable method mechanism; no similarity/quantization rescue sweep.

## Execution and integrity

    PYTHONPATH=shared:. timeout 300 /home/haipd/miniconda3/bin/python -m methods.information_probe.temporal_slot_inventory

CPU only, expected tens of seconds, hard300s timeout, progress every1000rows,
monitor30–60s while running. Output AS-C37-TEMPORAL-SLOTS_run.json, no overwrite.
Unit tests precede execution: padding exclusion, signed zero, adjacent and
nonadjacent repeats, FP16-only merger, repeated dense indices, equal streams
and distinct-stream fusion cancellation, finite rejection. No retries concealed.
Run report includes11/11fallacy scan. This is a diagnostic, not a GO pilot.
