# Changelog

## 0.1.0a1 — initial alpha

- Add feature-table input validation and trajectory/segment-aware grouping.
- Add explicit periodic-angle encoding and configurable feature scaling.
- Add PCA scores and explained-variance diagnostics.
- Add covariance-based tICA, implied-timescale reporting and VAMP-2 lag diagnostics.
- Add K-means, DBSCAN and optional HDBSCAN wrappers, cluster quality metrics, K-means seed stability and DBSCAN parameter sweeps.
- Add histogram free-energy surfaces, basic state-transition and implied-timescale diagnostics.
- Add optional PyTorch autoencoder baseline.
- Add synthetic demo, tests, docs, an example 12D MBD–chaperone feature spec, and CI workflow.

## 0.1.0a2

- Added strict keyed joining of cluster assignments to mechanistic frame-level observables.
- Added cluster-wise and trajectory-wise summaries for Cu-S distances, contact counts, RMSD, and user-defined event thresholds.
- Explicitly reject duplicate or mismatched frame identities instead of joining by row order.
