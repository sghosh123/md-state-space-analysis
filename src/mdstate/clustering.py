"""Cluster estimators and metrics that handle density-clustering noise explicitly."""
from __future__ import annotations

from collections.abc import Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score


def _validate_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must have shape (n_samples >= 2, n_features >= 1).")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")
    return X


def fit_kmeans(X: np.ndarray, n_clusters: int = 5, random_state: int = 42, n_init: int = 10) -> tuple[KMeans, np.ndarray]:
    X = _validate_X(X)
    if not 2 <= n_clusters < len(X) + 1:
        raise ValueError("n_clusters must be >= 2 and <= number of samples.")
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    return model, model.fit_predict(X)


def fit_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 10, metric: str = "euclidean") -> tuple[DBSCAN, np.ndarray]:
    X = _validate_X(X)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite value.")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    return model, model.fit_predict(X)


def fit_hdbscan(X: np.ndarray, min_cluster_size: int = 20, min_samples: int | None = None, metric: str = "euclidean"):
    """Fit optional HDBSCAN, importing its optional dependency only when requested."""
    X = _validate_X(X)
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError("HDBSCAN support requires `pip install 'md-state-space-analysis[density]'`.") from exc
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)
    return model, model.fit_predict(X)


def cluster_summary(labels: Iterable[int]) -> pd.DataFrame:
    labels = np.asarray(list(labels))
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence.")
    values, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    result = pd.DataFrame({"cluster_label": values, "n_frames": counts, "fraction": counts / n})
    result["is_noise"] = result["cluster_label"] == -1
    return result.sort_values(["is_noise", "n_frames"], ascending=[True, False]).reset_index(drop=True)


def score_clustering(X: np.ndarray, labels: Iterable[int], silhouette_sample_size: int = 10000, random_state: int = 42) -> dict[str, float | int]:
    """Compute cluster count, noise fraction and quality scores where defined.

    For DBSCAN/HDBSCAN, noise points (label -1) are excluded from silhouette and
    Davies–Bouldin calculations; those scores therefore describe assigned points only.
    They are not directly comparable to scores computed on all samples.
    """
    X = _validate_X(X)
    lab = np.asarray(list(labels), dtype=int)
    if len(lab) != len(X):
        raise ValueError("labels must have one label per row of X.")
    noise = lab == -1
    assigned_mask = ~noise
    unique_assigned = np.unique(lab[assigned_mask])
    out: dict[str, float | int] = {
        "n_samples": int(len(lab)),
        "n_clusters": int(len(unique_assigned)),
        "noise_fraction": float(noise.mean()),
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
    }
    xa, la = X[assigned_mask], lab[assigned_mask]
    if len(np.unique(la)) >= 2 and len(la) > len(np.unique(la)):
        try:
            kwargs = {}
            if silhouette_sample_size > 1 and len(la) > silhouette_sample_size:
                kwargs = {"sample_size": int(silhouette_sample_size), "random_state": int(random_state)}
            out["silhouette"] = float(silhouette_score(xa, la, **kwargs))
            # Davies–Bouldin remains computed on all assigned samples.
            out["davies_bouldin"] = float(davies_bouldin_score(xa, la))
        except ValueError as exc:
            warnings.warn(f"Could not compute all cluster metrics: {exc}", RuntimeWarning)
    return out


def compare_cluster_labels(labels_a: Iterable[int], labels_b: Iterable[int]) -> float:
    """Label-permutation-invariant adjusted Rand index for two partitions."""
    a, b = np.asarray(list(labels_a)), np.asarray(list(labels_b))
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Both label arrays must be one-dimensional and equal length.")
    return float(adjusted_rand_score(a, b))


def kmeans_parameter_sweep(
    X: np.ndarray,
    k_values: Iterable[int],
    seeds: Iterable[int] = (42, 43, 44),
) -> pd.DataFrame:
    """Summarize K-means sensitivity across cluster counts and random seeds.

    Returns one row per k with mean/std inertia, silhouette, Davies–Bouldin, and mean
    pairwise adjusted Rand index between seed-specific partitions. This is a stability
    diagnostic, not an automatic optimizer for the scientifically correct cluster count.
    """
    X = _validate_X(X)
    ks = sorted(set(int(k) for k in k_values))
    seed_values = list(dict.fromkeys(int(s) for s in seeds))
    if not ks or not seed_values:
        raise ValueError("Provide at least one k value and one seed.")
    if any(k < 2 or k >= len(X) for k in ks):
        raise ValueError("Every k must be >=2 and smaller than the sample count.")
    rows = []
    for k in ks:
        inertia_values, silhouette_values, db_values = [], [], []
        all_labels = []
        for seed in seed_values:
            model, labels = fit_kmeans(X, k, random_state=seed)
            metrics = score_clustering(X, labels)
            inertia_values.append(float(model.inertia_))
            silhouette_values.append(float(metrics["silhouette"]))
            db_values.append(float(metrics["davies_bouldin"]))
            all_labels.append(labels)
        pair_ari = [compare_cluster_labels(all_labels[i], all_labels[j])
                    for i in range(len(all_labels)) for j in range(i + 1, len(all_labels))]
        rows.append({
            "n_clusters": k, "n_seeds": len(seed_values),
            "inertia_mean": float(np.mean(inertia_values)), "inertia_std": float(np.std(inertia_values, ddof=1)) if len(inertia_values) > 1 else 0.0,
            "silhouette_mean": float(np.nanmean(silhouette_values)), "silhouette_std": float(np.nanstd(silhouette_values, ddof=1)) if len(silhouette_values) > 1 else 0.0,
            "davies_bouldin_mean": float(np.nanmean(db_values)),
            "seed_pairwise_ari_mean": float(np.mean(pair_ari)) if pair_ari else float("nan"),
        })
    return pd.DataFrame(rows)


def dbscan_parameter_sweep(
    X: np.ndarray,
    eps_values: Iterable[float],
    min_samples_values: Iterable[int],
) -> pd.DataFrame:
    """Evaluate a DBSCAN parameter grid without declaring a single best setting."""
    X = _validate_X(X)
    epses = sorted(set(float(v) for v in eps_values))
    mins = sorted(set(int(v) for v in min_samples_values))
    if not epses or not mins:
        raise ValueError("Provide at least one eps and one min_samples value.")
    rows = []
    for eps in epses:
        for min_samples in mins:
            _, labels = fit_dbscan(X, eps=eps, min_samples=min_samples)
            metrics = score_clustering(X, labels)
            rows.append({"eps": eps, "min_samples": min_samples, **metrics})
    return pd.DataFrame(rows)
