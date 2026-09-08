# Research methods

Each proposal method owns one directory containing its Python package, method-only configs,
campaign scripts, tests, and documentation. Reusable dataset, feature, evaluation, upstream bridge,
and runtime utilities live in `shared/`; vendored dependencies remain in `third_party/`.

Current methods:

- `elsc/`: Evidence-Localized Sign Contrast (proposal 1).
- `dive/`: Discriminative Visual Evidence Learning (proposal 2). Its package uses the
  `methods/dive/src/dive` layout so method-owned code remains isolated from shared utilities.
