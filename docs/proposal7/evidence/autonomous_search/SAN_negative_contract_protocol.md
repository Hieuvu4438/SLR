# SAN negative-generation contract check

2026-09-16. Source verification, academic-research-suite; AI-assisted.
Pinned SAN: `82aba9cbc1beb403abef6e9a3875ca52479805c8`.
This protocol is written before fixture execution. No dataset, checkpoint,
tokenizer download, training, DEV/TEST evaluation or upstream modification.

Question: what does the released generator actually guarantee? Extract only
`generate_hard_negatives`, `collate_fn` and `CrossEn` through Python AST, without
importing the upstream packages or constructing their dataset class.

Fixed fixtures, seed 42, five negatives:

1. One word with one non-self substitute: all five outputs must be identical,
   illustrating sampling with replacement, not an implementation error by itself.
2. A self-only table entry: all outputs equal the source. This table is deliberately
   adversarial; its presence in the authors' actual table is NOT asserted.
3. No table coverage and two distinct captions: fallback uses the other caption.
4. No coverage and two identical captions: empty fallback pool raises IndexError.
5. Two captions that are reversed word orders, no coverage: execute source collate
   with an explicitly mocked reverse-word swap and forced augmentation branch.
   Check that generated negatives equal the subsequently swapped positives.
   The mock is one possible permutation, not an execution of EDA or an estimate
   of EDA probabilities. A passthrough tokenizer records strings, not token IDs.
6. Check two-rank, two-example, five-negative ownership indexing with integer
   tags; reshape/diagonal selection should recover each example's own negatives.
   This tests indexing algebra, not actual distributed execution.
7. Actual CrossEn on six tied scalar logits: loss equals log(6); if all scores
   share the same scalar variable, its gradient is zero (tolerance 1e-12).
8. Repeated negative score at multiplicities one and five: actual CrossEn equals
   log(1 + m exp(n-p)); loss strictly increases with m (tolerance 1e-12).

These are conditional source guarantees, not population-frequency measurements,
semantic false-negative judgments, published-result reproduction or efficacy.
Successful assertions do not authorize a training run. Any empirical exposure
claim needs the actual table, preprocessing/tokenization and sampled TRAIN path.
Generic deduplication/validation is a reproduction control; local-hard-negative,
lexical-support and equivalence-positive method families remain closed.
