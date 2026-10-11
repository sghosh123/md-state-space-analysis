# Validation report — MD State-Space Analysis 0.1.0a1

## Summary

The alpha package passed **29 automated tests** in the build environment. The source was compiled, a wheel was built using the locally installed build backend without network-dependent isolation, the wheel was installed into an isolated target directory, and the installed wheel successfully ran the synthetic demo. The CLI workflows for PCA/tICA/clustering, tICA lag diagnostics, K-means/DBSCAN parameter sweeps, histogram FES and the optional autoencoder were exercised on synthetic data.

## Checks executed

| Check | Result | Scope |
|---|---|---|
| `pytest -q` | 29 passed | Feature transforms, periodic angle encoding, PCA, tICA, lag diagnostics, clustering, parameter sweeps, FES, transition diagnostics, CLI demo, optional autoencoder. |
| `python -m compileall -q src tests examples` | Passed | Python syntax/byte compilation. |
| `mdstate demo --outdir ...` | Passed | Creates four independent synthetic trajectories and writes PCA/tICA scores, cluster labels/metrics, tables and plots. |
| `mdstate tica-lags ... --lags 2,5,10` | Passed | tICA eigenvalues/timescales plus VAMP-2 scores over frame lags. |
| `mdstate cluster-sweep ... --mode kmeans` | Passed | K-means k/seed sensitivity and pairwise ARI stability output. |
| `mdstate fes ... --coordinates TIC1,TIC2` | Passed | 2D histogram FES table, plot and metadata from the synthetic tICA projection. |
| `examples/run_autoencoder.py` | Passed | PyTorch latent projection, held-out synthetic reconstruction, checkpoint serialization. |
| Wheel build | Passed | `md_state_space_analysis-0.1.0a1-py3-none-any.whl`. |
| Isolated wheel install + CLI demo | Passed | Import and demo through the installed wheel, independent of editable source import. |
| Archived 12D feature specification | Passed | End-to-end CLI test using synthetic data with the same column names and metadata layout. |

## Scientific sanity checks included

- Periodic-angle unit test confirms values at +179° and -179° remain near each other after sine/cosine encoding.
- Synthetic AR(1) process test checks the scalar tICA lag-3 eigenvalue against the theoretical autocorrelation `rho**3` within tolerance.
- Trajectory/segment splitting test checks frames are sorted within groups and segment boundaries remain separate.
- tICA test confirms a one-frame segment is excluded from lagged covariance fitting while still being projectable by a fitted model.
- FES tests verify probability mass normalization and minimum occupied-bin free energy equal to zero.

## Not validated by this report

1. **Production trajectory results:** no scientific analyses were run against the full original topology/trajectory set. The tests use synthetic feature tables.
2. **External tICA equivalence:** the covariance-based implementation has synthetic sanity tests, but has not yet been compared on identical data against deeptime/PyEMMA. Do that before using numerical tICA eigenvalues as publication results.
3. **HDBSCAN runtime:** the `hdbscan` package was not installed in the build environment; its optional import/error path exists, but HDBSCAN clustering itself was not executed.
4. **Autoencoder historical reproducibility:** the supplied segment archives contained an `AE_latent_FES.png` output but no matching training `.py`/`.ipynb` script. The PyTorch autoencoder included here is a new baseline, not a validated reproduction of the archived figure.
5. **MSM validity:** included transition matrices/implied-timescale utilities are basic diagnostics, not a replacement for a complete MSM framework or Chapman–Kolmogorov validation.
6. **License/authorship:** author/contact fields are placeholders and the license is unresolved. Replace placeholders and settle licensing before public release.
7. **Feature protocol:** the `yeast_mbd_chaperone_12d.json` spec reflects the archived generator's column names and apparent Å/degree conventions. Confirm exact production CSV headers and units before use.

## Release recommendation

The source package is suitable as an **alpha software portfolio draft**. Do not present its synthetic validation as proof of scientific conclusions. Before a stable release, validate the feature definitions against original trajectory data, compare tICA against a trusted implementation, test HDBSCAN in an environment with the optional dependency, record software versions/stride/physical units, and resolve licensing/authorship.
