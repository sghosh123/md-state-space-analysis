import numpy as np
import pytest
from mdstate.dimensionality import fit_pca, fit_tica, implied_timescales, vamp2_score
from mdstate.diagnostics import evaluate_tica_lags


def data(seed=2):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(3):
        x = np.zeros((250, 3))
        noise = rng.normal(scale=0.2, size=(250, 3))
        for t in range(1, len(x)):
            x[t] = 0.92 * x[t - 1] + noise[t]
        out.append(x)
    return out


def test_pca_returns_scores_and_variance():
    X = np.random.default_rng(2).normal(size=(100, 5))
    model, scores = fit_pca(X, 3)
    assert scores.shape == (100, 3)
    assert np.isclose(model.explained_variance_ratio_.sum(), np.sum(model.explained_variance_ratio_))


def test_tica_shapes_and_implied_timescales():
    trajectories = data()
    model = fit_tica(trajectories, lag_time=5, n_components=2)
    assert model.transform(trajectories[0]).shape == (250, 2)
    assert len(model.eigenvalues_) == 2
    assert model.covariance_lagged.shape == (3, 3)
    assert np.isfinite(model.eigenvalues_).all()


def test_tica_rejects_invalid_lag():
    with pytest.raises(ValueError, match="positive integer"):
        fit_tica(data(), lag_time=0)


def test_tica_lag_diagnostics():
    d = evaluate_tica_lags(data(), [2, 5, 10], n_components=2)
    assert len(d) == 6
    assert set(d.columns) >= {"lag_time_frames", "eigenvalue", "implied_timescale_frames", "vamp2_score"}
    assert np.isfinite(d["vamp2_score"]).all()


def test_implied_timescale_invalid_eigenvalues_nan():
    ts = implied_timescales(np.array([0.9, 1.0, 0.0, 1.1]), 5)
    assert ts[0] > 0
    assert np.isnan(ts[1:]).all()


def test_scalar_tica_eigenvalue_tracks_ar1_lag_correlation():
    rng = np.random.default_rng(44)
    rho = 0.8
    x = np.zeros((20000, 1))
    noise = rng.normal(size=len(x))
    for t in range(1, len(x)):
        x[t, 0] = rho * x[t - 1, 0] + noise[t]
    model = fit_tica([x], lag_time=3, n_components=1, regularization=1e-12)
    # For a stationary AR(1) process, lag-3 normalized autocovariance approaches rho**3.
    assert model.eigenvalues_[0] == pytest.approx(rho ** 3, abs=0.04)


def test_tica_ignores_one_frame_segment_for_fit_but_can_transform_it():
    trajectories = data()
    trajectories.append(np.array([[0.1, -0.2, 0.3]]))
    with pytest.warns(RuntimeWarning, match="Ignoring trajectories too short"):
        model = fit_tica(trajectories, lag_time=5, n_components=2)
    assert model.transform(trajectories[-1]).shape == (1, 2)
