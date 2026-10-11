import numpy as np
import pytest
from mdstate.clustering import (fit_kmeans, fit_dbscan, cluster_summary, score_clustering, compare_cluster_labels)


def data():
    rng = np.random.default_rng(3)
    return np.vstack([rng.normal(-2, 0.25, (60, 2)), rng.normal(2, 0.25, (60, 2))])


def test_kmeans_finds_two_synthetic_clouds():
    model, labels = fit_kmeans(data(), 2, random_state=17)
    assert len(np.unique(labels)) == 2
    assert score_clustering(data(), labels)["silhouette"] > 0.7


def test_dbscan_reports_noise():
    X = data()
    X = np.vstack([X, [10, 10]])
    model, labels = fit_dbscan(X, eps=0.7, min_samples=5)
    summary = cluster_summary(labels)
    score = score_clustering(X, labels)
    assert score["noise_fraction"] >= 1 / len(X)
    assert summary["n_frames"].sum() == len(X)


def test_cluster_score_label_length():
    with pytest.raises(ValueError, match="one label per row"):
        score_clustering(data(), [0, 1])


def test_adjusted_rand_is_label_permutation_invariant():
    assert compare_cluster_labels([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0


def test_kmeans_sweep_reports_seed_stability():
    from mdstate.clustering import kmeans_parameter_sweep
    result = kmeans_parameter_sweep(data(), [2, 3], seeds=[1, 2])
    assert len(result) == 2
    assert "seed_pairwise_ari_mean" in result
    assert result.loc[result["n_clusters"] == 2, "seed_pairwise_ari_mean"].iloc[0] > 0.9


def test_dbscan_sweep_reports_noise_fraction():
    from mdstate.clustering import dbscan_parameter_sweep
    result = dbscan_parameter_sweep(data(), [0.3, 0.8], [4])
    assert len(result) == 2
    assert "noise_fraction" in result
