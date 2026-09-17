# SAN TRAIN fallback-collision exposure screen

2026-09-16, before census execution. academic-research-suite/source verification.

Use only the pinned SAN `data/labels.train.cleaned` file (7096 rows verified
from container/schema inspection). Do not read DEV/TEST labels or load models.
Decode its primitive-only gzip pickle with a restricted unpickler denying class
resolution and persistent objects. No upstream dataset constructor/import.

Earlier C16 used English caption_model and CiCo tokens; it cannot provide this
German SAN exposure bound. No C16 model/data rerun.

Fixed definitions: raw strings and case/punctuation-preserving whitespace tokens,
matching the inspected textaugment random_swap path. A fallback candidate must
have raw string different from the source, as SAN explicitly filters it.
Its raw string must equal its space-joined tokens to match EDA output exactly.
Inventory (a) arbitrary token-permutation reachability and (b) zero or one
transposition reachability, the latter corresponding to the inspected default
n=1 EDA implementation. Zero transpositions includes no-op/failure and equal-word
swaps. Verify reachability helper against exhaustive swaps on small toy inputs.

Record eligible source-row counts, ordered source/target-row counts, group counts,
largest identical-string multiplicity, whitespace normalization changes, and
per-eligible-row counterpart counts. Do not assign relevance or semantic labels.

Under explicitly assumed uniform, without-replacement local batches B=64,
N=7096 and K_i reachable counterpart rows, probability of any counterpart in a
batch is 1-product_{j=0}^{62}(1-K_i/(N-1-j)). Half its average is a conservative
model-based upper bound for an augmented fallback collision, allowing every row
to take fallback and every available counterpart to be selected/reached. This is
NOT a measured training rate. Actual table coverage, negative draws and exact
swap probabilities can only reduce this pretokenization bound under these
assumptions. Tokenizer normalization/truncation collisions are outside scope.

Check whether max identical-string multiplicity is below64: if so an all-identical
full local batch cannot occur without replacement, irrespective of table coverage.
Source uses drop_last=True; default four ranks evenly divide7096. Do not generalize
to other batch sizes, custom samplers or other world sizes.

Textaugment is absent locally and unpinned/omitted in SAN requirements. Inspected
upstream implementation:8553ea0a4dd5701df7f4c5f17c9b315ff099f11d, eda.py
SHA256835ecd37f8ea84f489743a430ff878ae7a66c1f0def83f7bda78e742a188107a.
This is a conditional source match, not verified author runtime version.

No significance test or SOTA/investment threshold. This bounds one previously
demonstrated failure case; it does not authorize closed-family repair training.
