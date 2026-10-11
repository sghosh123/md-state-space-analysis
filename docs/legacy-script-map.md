# Map from archived research scripts to reusable package functions

The source archives supplied for review contain project-specific scripts under the yeast MBD/chaperone trajectory-analysis tree. This package refactors reusable method patterns; it does **not** blindly copy the archived scripts because many contain hard-coded file names, plotting code intertwined with analysis, and implicit feature-column assumptions.

| Archived script / workflow | Archive segment | Intended reusable destination | Notes for migration |
|---|---:|---|---|
| `kmeans_cluster_angles_distances.py` | 06 | `mdstate.clustering`, `mdstate.dimensionality`, CLI | Original uses `df.columns[1:-1]` as features, which can include unintended columns or drop a legitimate feature depending on table layout. New CLI recommends explicit `--features`. |
| `multi-k_means_cluster_angles_distances.py` | 07 | `kmeans_parameter_sweep` | New helper compares k values and random seeds, and reports pairwise adjusted Rand stability as well as inertia/silhouette. |
| `dbscan_auto_cluster_angles_distances.py` | 07 | `fit_dbscan`, `score_clustering` | `eps` and scaling are data-dependent; report noise fraction and avoid calling an automatically chosen parameter objectively optimal. |
| `tune_dbscan_params_with_sizes.py` | 10 | `dbscan_parameter_sweep` | New helper records cluster count, noise share, and scores for each eps/min_samples combination; inspect ranges and do not treat one score as ground truth. |
| `vector_based_angle_distance_analysis.py` | 10 | upstream feature table; periodic handling in `mdstate.preprocessing` | Hard-coded topology/trajectory file lists and residue selections are system-specific. Keep feature definitions in versioned configuration. |
| `tICA_lag_converge_on_slt-brg_pairs.py` | 07 | `mdstate.diagnostics.evaluate_tica_lags` | Confirm time-lag units and ensure pairs do not cross file/trajectory boundaries. Compare timescale trends, not only visual flatness. |
| `tICA_FE_on_vector_angle_distance_features_kJ_mol.py` | 08 | tICA projection plus downstream FES visualization | Projection, density estimation, and free-energy normalization should be separate steps with documented temperature and units. |
| `tICA_FE_on_vector_features_kJ_mol_sltbrdg_count.py` | 07 | tICA plus metadata/observable overlays | Salt-bridge counts can be used for interpretation/coloring; they should not leak into the feature matrix unless explicitly intended. |
| `tICA_FE_on_vector_features_kJ_mol_sltbrdg_count_basin_id.py` | 09 | state/cluster overlay (future extension) | Basin labels and post hoc Cu–S/contact labels are outcomes, not default input features. |
| `msm_on_vector_angle_distance_pyemma_tutorial.py` | 07 | `mdstate.kinetics` diagnostics; external MSM package for production | The source contains hard-coded trajectories and a PyEMMA-specific workflow. This package currently supplies only transition/timescale primitives, not a replacement for PyEMMA/deeptime validation. |
| `vector_based_angle_distance_analysis.py` | 10 | feature provenance / upstream generator | Segment identity and local frame number must both be retained when combining split trajectory files. |
| PCA in `kmeans_cluster_angles_distances.py` | 06 | `mdstate.dimensionality.fit_pca` | Existing PCA is used mainly for visualization; the package keeps PCA scores and explained variance explicit. |
| `AE_latent_FES` output image | 06 archive output inventory | `mdstate.autoencoder` (new baseline only) | No matching source `.py` or `.ipynb` was found in the supplied ZIP archives. The new autoencoder is therefore a new example implementation, not a port of the prior work. |

## Important distinction

Archive segments 01–05 were single large CSV data files. Source scripts relevant to this repository were located primarily in segments 06–10. This map reflects filenames and code inspected from those archive members; it does not imply every output figure or historical numerical result has been reproduced.
