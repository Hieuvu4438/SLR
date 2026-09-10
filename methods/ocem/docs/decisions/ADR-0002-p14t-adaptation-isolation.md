# ADR-0002: Isolate P14T adaptation from the released H2S route

- Status: Accepted
- Date: 2026-09-10
- Scope: WP-04 feature provenance

## Context

The local cache `ph_domain_aware_h2s_transfer_gpu` is complete, but its
checkpoint is the released How2Sign-adapted I3D. OCEM requires target-domain
adaptation from the selected dataset's train IDs only.

The pinned CiCo P14T trainer path is incomplete as released. Its P14T dataset
loader expects `misc/phoenix2014T/train_val_info.json`, which is absent, reads
the H2S class list, and hard-codes `../bsl/pseudo_from_i3d`. The supplied
pseudo-label launcher is How2Sign-specific and launches both train and test.

## Decision

Do not accept, rename, or silently mix the H2S-transfer stream as P14T target
adaptation. Build a manifest-scoped P14T adaptation input from the 7,096 locked
train IDs, record vocabulary/threshold/sampling/optimizer fields, and assert
that validation and test IDs are absent before training starts.

The upstream source remains read-only. Compatibility code belongs inside the
isolated OCEM package and must preserve the observed I3D preprocessing and
checkpoint-loading semantics with explicit missing/unexpected-key reports.

## Consequences

The domain-agnostic cache can be reused after its successful audit, saving one
full extraction pass. WP-04 and the P14T resource gate remain incomplete until
the adapted checkpoint and a window-aligned adapted cache are produced and
locked. This is a provenance requirement, not a claim that H2S transfer is an
invalid scientific ablation.
