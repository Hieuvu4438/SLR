# AS-C16 — exact-lexicon controls do not supply a relational probe

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-15
- Verification Status: ANALYZED (inventory and frozen pair scores)
- Version Label: AS-C16-result-v1

Completed exit0 in1.86s, protocol recorded before execution. Full provenance,
every caption/index and all61 scored pairs are in `AS-C16-LEXICAL_run.json`.
No fitting, synthetic data, changed positives, test access or new benchmark.

Both whole-word multiset and deployed BPE multiset searches independently find
the SAME2 train groups,19 rows and61 different-input pairs; neither finds a dev
group. Token multiplicity is preserved, punctuation ignored for word keys,
actual uniform content-token selection used for BPE keys.

|Train group|Rows|Different-order pairs|Distinct surface wordings|Inferred source prefixes|
|---|---:|---:|---:|---:|
|Placement of “now” in an evening farewell|17|60|2|17|
|Placement of “usually” in a forecast|2|1|2|2|

The first alternatives are “Now I wish you a nice evening” and “I wish you a
nice evening now”. The second moves “Usually” from before “the sun” to after
it in “the sun … shines in the next few days and it gets hot again”.

**Text-only interpretation:** these are adverb-placement alternatives, not clear
swaps of entities, events, arguments or bound quantities. Contextual/pragmatic
differences remain possible. No signed clips or expert language annotations
were inspected to establish whether any pair is a semantic minimal contrast.
They must not be relabeled equivalent or used as hard relational negatives.

All61 pairs have positive margins in ALL FOUR crossed retrieval directions,
well above the fixed1e-4 numerical band. Minimum margins: T2V_a5.923645,
T2V_b4.235100,V2T_a5.528214,V2T_b4.630531. These are raw frozen-model logits,
not percentages or full-gallery R1. Pair endpoints a/b use ascending row index,
not linguistic role. There are only TWO wording contrasts, not61 independent
linguistic examples;60 arise from repeated realizations of the first contrast.

Bounded positive inference: on these in-sample examples the deployed pipeline
is not exactly a bag-of-words function of the text. A permutation-invariant
function of the same content-token multiset would assign equal scores to both
texts for a fixed video, contrary to the measured strict V2T margins. This does
NOT establish relational grounding, out-of-sample generalization or semantic
correctness of paired-ID distinctions. It supplies no unaddressed failure from
which to derive a new relational architecture.

Decision: do not train a new Q09 relational probe on this inventory. Natural
exact-lexicon matched supervision is inadequate under this definition. Near-
lexicon compositional contrasts and validated sign-language relations remain
unmeasured; this is not a global information ceiling or exhaustion claim.

## Statistical checks (11/11)

1. Simpson: both token definitions and train/dev inventories disclosed; no
   subgroup performance generalization.
2. Ecological: no signer/linguistic claim from grouped caption counts.
3. Berkson: exact lexical equality is a narrow filter, not all compositional pairs.
4. Collider: no conditioned causal estimate.
5. Base rate:7096 train/519 dev screened;19 train rows/0 dev rows qualify.
6. Regression to mean: no correction gain claimed on selected errors.
7. Survivorship: all groups and61 pair results retained; none dropped by margin.
8. Look elsewhere: fixed two keys, no dev tuning or significance claim.
9. Forking paths: textual interpretation follows inventory and is explicitly
   qualitative; no hidden new selector or relabeling.
10. Correlation/causation: order-sensitive scoring is not evidence of correctly
    grounded sign-language relations.
11. Reverse causality: train separability cannot explain why the checkpoint
    learned these distinctions or whether they should be preserved.

Two new unit tests pass. Full experimental rerun not claimed. No method GO.
