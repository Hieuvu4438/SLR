# SAN: negative-generation guarantees and their limits

2026-09-16. academic-research-suite, source fact-check inline. ANALYZED;
AI-assisted, no independent human or linguistic review. No method GO.

## Verified source behavior

SAN revision `82aba9cbc1beb403abef6e9a3875ca52479805c8` is unchanged locally.
Read scope: datasets.py through collate_fn; utils.py CrossEn/AllGather;
models.py imports and SLRCLIP gathering/forward; training-loss call sites,
train.bash and README. No claim of a complete repository or paper audit.

[Pinned generator](https://github.com/joonmy/SAN/blob/82aba9cbc1beb403abef6e9a3875ca52479805c8/datasets.py#L114)
samples substitutions independently for each negative. With no covered words,
it samples other batch captions with replacement. It neither deduplicates
outputs nor validates output inequality. The collator generates negatives
before optionally applying EDA random_swap to positives (lines177–190).
The supplied launcher requests five negatives and a user-supplied table path.

[Protocol](SAN_negative_contract_protocol.md),
[executable](../../../../methods/information_probe/san_negative_contract.py),
[retained output and source fingerprints](SAN-NEGATIVE-CONTRACT.json).
Eight fixed synthetic fixtures passed in two executions, both exit0, approximately
.843/.851s. The second output is retained. No upstream imports, real tokenizer,
EDA execution, checkpoint, data, GPU, DEV/TEST evaluation or optimizer updates.
Existing information-probe regression suite:90passed in1.81s, exit0. The eight
standalone fixture checks are separate from that pytest count. git diff --check
exited0; new untracked artifacts are not covered by that tracked-diff check.

| Check | Result | Interpretation boundary |
|---|---|---|
| One available substitute | Five identical negative strings | Replacement sampling; not intrinsically a bug |
| Deliberately self-only table | Five unchanged strings | No self-entry guard; actual table contents unknown |
| No coverage, two distinct captions | Other caption repeated five times | Fallback is not necessarily visually hard |
| No coverage, all captions identical | IndexError | Conditional empty-pool failure; frequency unknown |
| Mocked reverse-word swap after fallback | Positive equals all five associated negatives | Source ordering admits collision; not actual EDA frequency |
| Two-rank ownership tags | Correct negative owner recovered | No indexing bug in tested algebra; distributed runtime not tested |
| Six equal, shared scalar logits | CE = log(6) = 1.7917594692; shared gradient ≈1.67e−16 | Inseparable deterministic copies provide no discriminative gradient |
| One versus five repeated negative logits | CE .3132616875 versus1.0435917782 | Multiplicity changes weight at fixed scores |

For one positive score p and m identical negative scores n, hard CE is
`log(1 + m exp(n-p))`. Thus duplicate sampling behaves like an additive log(m)
shift to that negative's aggregate contribution. This is elementary loss
algebra, not a novel method or evidence that deduplication improves retrieval.
When copies share the same score function and input, their gradients cancel
in the all-identical case. Independent dropout can break exact score equality;
the fixture does not measure SAN parameter gradients under training stochasticity.

The augmentation fixture uses a forced branch, a reversal mock and a string-
passthrough tokenizer. It proves the collator does not prevent this collision,
not that the EDA package produces it for a particular real example. No semantic
judgment about any PH caption is made. Token-level collisions were not measured.

## Primary-source check and evidence quality

The [official ACL2026 record](https://aclanthology.org/2026.acl-long.1302/)
identifies SAN as visually confusable negative mining and reports fine-grained
improvements with preserved coarse retrieval. Read scope: metadata/abstract only;
these reported outcomes were not reproduced or used for experiment selection.
This is counterevidence to any blanket claim that SAN's negative supervision
does not work. The browser fetch of the pinned GitHub file failed; code claims
instead rely on the locally inspected revision and recorded file hashes.

Evidence grade: primary source plus directly executed conditional checks is
strong evidence for these narrow code behaviors, but insufficient for empirical
retrieval harm. No population study level or independent replication claimed.
Author-associated code is not independent corroboration of published efficacy.
Journal/COI investigations are outside this bounded source audit; no bibliographic
API or DOI-resolution verification was performed or claimed.

## Decision

Do not launch a deduplication, swap-removal, hard-mining or soft-positive campaign
from these fixtures. Validation/empty-pool handling would be reproduction repairs,
not a novel SOTA contribution. Lexical support, generic local-hard-negative and
equivalence-positive mechanisms remain closed by the registry.

An exposure claim would require the actual negative table, its provenance,
real preprocessing/tokenization/EDA behavior, and a permitted TRAIN-only sampling
trace. These fixtures cannot supply those inputs. No claim that every public
resource was searched in this turn. Do not substitute an invented table and call
its measured frequency an author-method defect. No new Q38 or Proposal8; broader
research remains open. Do not repeat these fixtures as another discovery cycle.
