# Validation report — MD State-Space Analysis 0.1.0a2

The existing PCA/tICA/clustering pipeline was extended with strict cluster-to-mechanism association in this alpha update.

## Current checks

- 35 automated tests pass in the build environment.
- The Python wheel builds successfully from `pyproject.toml`.
- New tests cover exact keyed joins (including differently named identity columns across repositories), shuffled row order, duplicate frame keys, mismatched frame sets, explicit event thresholds, and replicate-aware output tables.
- The CLI writes frame-level joined data, cluster-level descriptive summaries, and per-trajectory/per-cluster summaries.

## Remaining research validation

These checks do not establish scientific validity of an upstream cluster assignment, cutoff, tICA lag, or MSM. Validate real production tables and preserve trajectory/segment/local-frame keys. Mechanism tables must be pre-aggregated to one row per frame before association; do not arbitrarily select a row from long-form event data. Do not treat correlated frames as independent replicates.

The package license and author/citation details remain to be resolved before a public stable release.
