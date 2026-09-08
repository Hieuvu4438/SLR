# DIVE-SLR v2 baseline adaptations

No SEDS source patch has been applied yet. Real adapter/reproduction work remains blocked until the
pinned SEDS snapshot and compatible resources are inventoried. The following required adaptations
from the implementation spec remain open and must be recorded with exact patches when implemented:

- true train/dev/test routing and dev-only checkpoint selection;
- candidate padding masked before softmax;
- pure/prelogit fusion scoring parity with native directional mix;
- score orientation and stable-ID tie handling;
- pre-global-Transformer RGB/pose feature taps and explicit raw-time receptive fields.

The existing CiCo compatibility work for proposal 1 is not silently reused as a SEDS adaptation.
