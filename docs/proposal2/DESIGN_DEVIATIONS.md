# DIVE-SLR v2 design deviations

## D-001 — method-owned source root

- Spec clause: §3.1 illustrates `src/dive`, `configs`, `tests`, and `docs` at repository root.
- Implementation: DIVE-specific files live under `methods/dive/{src,configs,tests,docs}` and the
  root package configuration discovers `methods/dive/src`.
- Reason: this repository hosts multiple proposal methods. The existing project boundary places
  each method under `methods/`, while only genuinely reusable infrastructure belongs in `shared/`.
- Impact: installed imports and CLI remain `dive.*` / `dive`; source paths are prefixed by
  `methods/dive/`. There is no change to the method, tensor contracts, or experiment protocol.
- Resolution: accepted repository-layout adaptation; keep future DIVE-only code in this boundary.

No scientific-method deviations are currently active.
