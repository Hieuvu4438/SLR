# SAN TRAIN exposure: two conditional failure cases do not support a repair campaign

2026-09-16. academic-research-suite/source verification, inline; AI-assisted.
ANALYZED, no independent human or linguistic review. No method GO.

## Outcome

The previous [synthetic source checks](SAN_negative_contract_result.md) were valid
conditional examples. New evidence limits their applicability to the supplied
SAN TRAIN data and launcher:

- **Zero** of7096 German TRAIN captions can become a distinct other TRAIN caption
  through zero/one word transposition under the inspected EDA string operations.
  Thus that fallback/augmentation exact-string collision cannot occur here under
  these conditions. No real augmentation RNG trace was executed.
- The largest identical raw-caption group has **63** rows. An all-identical local
  batch of64 cannot occur without replacement. The supplied four-rank launcher
  evenly partitions7096 examples, and the loader drops incomplete local batches.
  Thus the empty-fallback-pool exception is not reachable with this configuration.
- Allowing arbitrary permutations instead produces114 eligible source rows
  (1.606539%),5552 ordered source/target row pairs across4 multiset groups.
  This deliberately broader support does not describe the inspected default
  single-swap augmentation. Under a uniform B64 sampling model, half the mean
  counterpart-cooccurrence probability is0.278666%—a loose upper bound on row
  exposure, **not** a measured collision rate or possible retrieval improvement.

The primary decision is to **deprioritize these two reproduction-repair routes**.
There is no justification for training a swap-removal or empty-pool repair arm.
This does not clear table-generated negatives, tokenizer collisions, duplicate
weighting or semantic false negatives; those require different evidence.

## Sources and execution

[Protocol](SAN_train_collision_protocol.md),
[script](../../../../methods/information_probe/san_train_collision.py),
[census and fingerprints](SAN-TRAIN-COLLISION.json),
[independent validation](SAN-TRAIN-COLLISION-validation.json).

Read only `third_party/SAN/data/labels.train.cleaned`, from SAN commit
`82aba9cbc1beb403abef6e9a3875ca52479805c8`. TRAIN SHA256:
`4e1927bf1821b4f634b0503fe2747d0b0074b7cd1722336249768fb5b139d2b5`.
Primitive pickle opcodes were screened before loading with class/persistent-object
resolution forbidden. No upstream constructor, model, tokenizer or EDA import.
All7096 strings were already whitespace-normalized. No relevance labels changed.

The reachability helper passed340 exhaustive toy pair checks. Two census
executions exited0, with identical saved values. A posthoc independent check
enumerated769199 actual TRAIN transpositions, performed exact raw-string lookup,
and found zero matching distinct captions (exit0,.197s). This independent check
validates the zero-hit result, not the entire probability model or sampler.

Earlier [C16](AS-C16_result.md) used English caption_model and CiCo BPE, so its
counts were **not** reused as German SAN exposure. Only that historical result
and its code were read; no C16 scoring/cache/model experiment was repeated.

[Textaugment source](https://github.com/dsfsi/textaugment/blob/8553ea0a4dd5701df7f4c5f17c9b315ff099f11d/textaugment/eda.py)
was fully read at commit8553ea0a4dd5701df7f4c5f17c9b315ff099f11d.
`random_swap` defaults to n=1, uses case-preserving whitespace split, calls
`swap_word`, and rejoins with spaces. Its helper may return unchanged words.
Source SHA256835ecd37f8ea84f489743a430ff878ae7a66c1f0def83f7bda78e742a188107a.
SAN requirements omit textaugment and the local environment lacks it. Therefore
this is a conditional match to inspected dependency code, **not verified author
runtime provenance**. No package/corpus installation or downloads occurred.

The supplied SAN loader at train_vlp_v2.py:179 uses a shuffled DistributedSampler;
lines183–190 set the local batch size and drop_last=True. train.bash specifies
four ranks and batch64. The probability bound assumes uniform without-replacement
cohort membership rather than asserting actual frequencies from a particular seed.

## Negative-table resource check

Read-only GitHub API checks verified both public branch trees, untruncated:

| Branch | Commit | Files | Relevant resource evidence |
|---|---|---:|---|
| main |82aba9cbc1beb403abef6e9a3875ca52479805c8|16|Supplied labels and source, no negative table; README says Coming Soon |
| master |2572f62b24f732678368b7377222789a9603ec74|10|Source/configs only; launcher points to an author's local train_hard_word_777.pkl.gz |

Tags and releases are empty lists, with no pagination Link header. Branches also
have no pagination Link. No compatible negative-table artifact was found in
these checked branches/releases. This is not a claim about private, deleted,
unlinked or separately hosted resources. No issue comments were read, no author
was contacted, and no new mining table was invented as a substitute.

API endpoints: [branches](https://api.github.com/repos/joonmy/SAN/branches?per_page=100),
[releases](https://api.github.com/repos/joonmy/SAN/releases?per_page=100),
[tags](https://api.github.com/repos/joonmy/SAN/tags?per_page=100),
[main tree](https://api.github.com/repos/joonmy/SAN/git/trees/82aba9cbc1beb403abef6e9a3875ca52479805c8?recursive=1),
[master tree](https://api.github.com/repos/joonmy/SAN/git/trees/2572f62b24f732678368b7377222789a9603ec74?recursive=1).
Browser API opens failed; direct read-only HTTP returned the inspected JSON.

Response SHA256s, in the endpoint order above:

```text
2e932f3c864eb71ea982e5abd0deb118c24843c8f901b7c8c82b78eaa01ae0dd
4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945
4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945
8d9074692279ab72be500ebd7ff9c7e39700c0ff6a6883c375344209890ac741
af68681df32fef64ffa7d8823c7ff4a3b961bb3b02e8e1001f08ebefd607dc9f
```

The alternate branch's launcher/config were read completely; they provide local
paths, not download links. Their hashes are31ed87b6eade4599b89f46dd64d9fc86952650c6ebc2c7b6ea810fba6ab58162
and8c5c0fc43453aed0f551536e426c84823005a6c851c1ba1d99539085f8001a46.
Branch tree filenames include DEV/TEST label paths; no such label content was read.

## Evidence limits and next decision

These source/data checks are strong evidence for narrow string reachability and
resource availability within their stated scope, not a general assessment of
SAN's published effectiveness. They establish neither a global research barrier
nor a new method. No p-value, SOTA comparison, novel theorem or human annotation.
Tokenizer normalization/truncation may introduce other collisions; no exposure
claim is made without an actual tokenizer/table/runtime match.

No training, GPU, checkpoint, DEV/TEST label/evaluation access or upstream edits.
No Proposal8 or Q38. Lexical/support/mining/equivalence closures remain binding.
Two SAN source/exposure turns now have concrete outcomes; move to a different
mechanism rather than repeat the fixture/census, tune augmentation, recreate a
table or repeat resource queries without changed evidence. The overall goal
remains active, with no global blocker established.
