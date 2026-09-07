# ELSC method package

This directory is the canonical home of proposal 1's method-specific implementation:

- `elsc/`: importable method package (keeps the public Python name `elsc`);
- `configs/`: ELSC-Min, ELSC-Full, and method ablation templates;
- `scripts/`: method experiment campaigns and gates;
- `tests/`: tests whose assertions are specific to ELSC.

Repository-level `configs/`, `scripts/`, and the small top-level `elsc` package contain shared
assets or compatibility entry points while active historical commands migrate. New methods should
be peers of this directory and import reusable primitives from `shared/slr_common`, never from
another method package.
