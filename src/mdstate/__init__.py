"""Tools for reproducible molecular-dynamics state-space analysis."""

__version__ = "0.1.0a2"

from .clustering import cluster_summary, fit_dbscan, fit_kmeans, fit_hdbscan, score_clustering, kmeans_parameter_sweep, dbscan_parameter_sweep
from .dimensionality import fit_pca, fit_tica, implied_timescales
from .preprocessing import encode_features
from .fes import estimate_fes

__all__ = [
    "__version__", "encode_features", "fit_pca", "fit_tica", "implied_timescales",
    "fit_kmeans", "fit_dbscan", "fit_hdbscan", "score_clustering", "cluster_summary", "kmeans_parameter_sweep", "dbscan_parameter_sweep", "estimate_fes",
]
