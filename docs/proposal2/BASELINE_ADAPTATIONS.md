# DIVE-SLR v2 baseline adaptations

The official SEDS repository is now available at the exact detached commit
`434e3f714fcb6a7d1f4001fb9a246bbd93ec0246`. No SEDS source patch has been applied yet. Real
adapter/reproduction work remains blocked on compatible checkpoint/features. The following required
adaptations remain open and must be recorded with exact patches when implemented:

- true train/dev/test routing and dev-only checkpoint selection;
- candidate padding masked before softmax;
- pure/prelogit fusion scoring parity with native directional mix;
- score orientation and stable-ID tie handling;
- pre-global-Transformer RGB/pose feature taps and explicit raw-time receptive fields.

The existing CiCo compatibility work for proposal 1 is not silently reused as a SEDS adaptation.
