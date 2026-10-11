# Methodological notes and validation checklist

## Feature engineering

Record exact definitions for every feature, including residue/atom selections, units, periodic boundary treatment, and whether angles are in degrees or radians. The analysis CLI only applies periodic encoding when the corresponding angle columns are explicitly listed. Distance names ending in `_nm` or `_A` are conventions, not automatic unit conversions.

Binary contact/salt-bridge observables and continuous distances can be combined, but equal-variance scaling is a choice—not a neutral operation. Consider sensitivity analyses with separate feature blocks or scientifically motivated weights. Exclude identifiers and labels from the features, and ensure filtering (such as Cu–S distance cutoffs) is disclosed because it changes the sampled ensemble.

## PCA

PCA maximizes variance in the supplied transformed feature space. It is useful for compact visualization and identifying large sources of variability, but its leading components need not be the slowest dynamical coordinates. Report explained variance and avoid treating a two-dimensional projection as a complete representation of the landscape.

## tICA implementation

The package estimates centered instantaneous and lagged covariance matrices across all supplied trajectory segments, and solves a regularized generalized eigenproblem. Lagged pairs are formed within each supplied trajectory array only. Eigenvectors are normalized under the regularized instantaneous covariance metric. Eigenvalues are sorted by absolute magnitude.

The provided implied-timescale formula is `-lag / log(abs(lambda))` for `0 < abs(lambda) < 1`; values outside this range are undefined/invalid for that conventional interpretation and are returned as NaN. Use the frame lag multiplied by the actual saved-frame interval to convert frame units into physical time. Check sensitivity to regularization and features, and compare implied timescales across lags. Finite-sample estimates can exceed one in magnitude.

The current tICA estimator is a compact covariance implementation designed for auditable small/medium feature tables. It should be checked against a trusted implementation (for example, deeptime) for a representative dataset before publication of scientific results, particularly for nearly collinear features, strong unequal trajectory lengths, and non-equilibrium sampling.

## Clustering

K-means assumes compact, roughly Euclidean clusters and is sensitive to scaling and k. DBSCAN is sensitive to `eps` and `min_samples`; HDBSCAN may be useful for heterogeneous density but is an optional dependency. Compare parameter sensitivity, feature sets, and cluster partitions rather than relying only on an elbow plot or one silhouette score.

For density clustering, `-1` denotes noise. The reported silhouette/Davies–Bouldin scores are computed on non-noise observations if at least two non-noise clusters exist. Consequently, these scores should not be compared naively against scores calculated on all samples or different noise fractions.

## Transition/kinetic diagnostics

The transition helper counts transitions between supplied state labels at a chosen lag without crossing trajectory arrays and can symmetrize counts. States with no observed outgoing transitions are left as zero rows, not silently interpreted as absorbing states. These are diagnostics, not a validated Markov state model. Kinetic claims require justified featurization, state discretization, sampling, lag-time checks, implied-timescale convergence, and Chapman–Kolmogorov validation.

## Autoencoders

No matching autoencoder training source script or notebook was identified in the supplied ZIP segments; only an output image with an `AE_latent_FES`-style name appeared in the archive inventory. The optional PyTorch implementation here is a new baseline and should not be described as a direct port of the prior scientific model. Use held-out trajectory/replica data for validation. For latent collective-variable claims, reconstruction loss is not enough: test robustness and dynamical usefulness.

## Release checklist before research use

- [ ] Confirm feature columns and their physical meaning from the production table.
- [ ] Confirm trajectories/segments and frames are identified correctly; document the output stride.
- [ ] Test tICA eigenvalues/components against a trusted package on the same small dataset.
- [ ] Establish sensible feature scaling and periodic-angle conventions.
- [ ] Compare cluster assignments over seeds and parameter settings.
- [ ] For any MSM, validate implied timescales and Chapman–Kolmogorov behavior with an appropriate tool.
- [ ] For autoencoders, use trajectory-level validation and save model/preprocessing configuration.
- [ ] Add software version information and random seeds to the analysis manifest.
- [ ] Replace metadata placeholders and settle licensing before making the repository public.
