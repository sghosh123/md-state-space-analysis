"""PCA and time-lagged independent component analysis (tICA)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA


@dataclass
class TICAResult:
    mean_: np.ndarray
    eigenvalues_: np.ndarray
    components_: np.ndarray  # (n_components, n_features)
    lag_time: int
    covariance_0: np.ndarray
    covariance_lagged: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1 or not np.isfinite(X).all():
            raise ValueError("Transform input must be a finite 2D array with at least one row and feature.")
        if X.shape[1] != self.components_.shape[1]:
            raise ValueError("Input feature dimension does not match fitted tICA model.")
        return (X - self.mean_) @ self.components_.T

    @property
    def implied_timescales(self) -> np.ndarray:
        vals = np.abs(self.eigenvalues_)
        out = np.full(vals.shape, np.nan, dtype=float)
        mask = (vals > 0.0) & (vals < 1.0)
        out[mask] = -float(self.lag_time) / np.log(vals[mask])
        return out


def _validate_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Expected a two-dimensional (frames, features) array.")
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("At least two frames and one feature are required.")
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinite values.")
    return X


def fit_pca(X: np.ndarray, n_components: int = 2, whiten: bool = False) -> tuple[PCA, np.ndarray]:
    """Fit PCA and return ``(model, scores)``."""
    X = _validate_matrix(X)
    if not isinstance(n_components, int) or not 1 <= n_components <= min(X.shape):
        raise ValueError(f"n_components must be between 1 and {min(X.shape)}.")
    model = PCA(n_components=n_components, whiten=whiten, svd_solver="full")
    scores = model.fit_transform(X)
    return model, scores


def _normalize_trajectories(trajectories: Sequence[np.ndarray], lag_time: int) -> tuple[list[np.ndarray], np.ndarray, int]:
    if not isinstance(lag_time, int) or lag_time < 1:
        raise ValueError("lag_time must be a positive integer number of frames.")
    arrays = []
    for item in trajectories:
        arr = np.asarray(item, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError("Each trajectory must be a 2D array with at least one frame and feature.")
        if not np.isfinite(arr).all():
            raise ValueError("Trajectory features contain NaN or infinite values.")
        arrays.append(arr)
    if not arrays:
        raise ValueError("At least one trajectory is required.")
    dimensions = {a.shape[1] for a in arrays}
    if len(dimensions) != 1:
        raise ValueError("All trajectories must have the same number of features.")
    viable = [a for a in arrays if len(a) > lag_time]
    if not viable:
        raise ValueError("No trajectory has more frames than the requested lag_time.")
    if len(viable) != len(arrays):
        warnings.warn("Ignoring trajectories too short for the requested lag_time.", RuntimeWarning)
    mean = np.concatenate(viable, axis=0).mean(axis=0)
    n_pairs = sum(len(a) - lag_time for a in viable)
    return viable, mean, n_pairs


def fit_tica(
    trajectories: Sequence[np.ndarray],
    lag_time: int = 1,
    n_components: int = 2,
    regularization: float = 1e-8,
) -> TICAResult:
    """Fit a covariance-based tICA model without creating cross-trajectory lag pairs.

    ``trajectories`` must already be feature encoded/scaled consistently. Lag time is in
    frames, not ps/ns. ``regularization`` is added to the instantaneous covariance diagonal
    to stabilize generalized eigenvalue estimation when features are collinear.
    """
    if regularization < 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative.")
    arrays, mean, n_pairs = _normalize_trajectories(trajectories, lag_time)
    n_features = arrays[0].shape[1]
    if not isinstance(n_components, int) or not 1 <= n_components <= n_features:
        raise ValueError(f"n_components must be between 1 and {n_features}.")

    c0 = np.zeros((n_features, n_features), dtype=float)
    ctau = np.zeros_like(c0)
    for arr in arrays:
        centered = arr - mean
        x0, x1 = centered[:-lag_time], centered[lag_time:]
        c0 += 0.5 * (x0.T @ x0 + x1.T @ x1)
        ctau += 0.5 * (x0.T @ x1 + x1.T @ x0)
    c0 /= n_pairs
    ctau /= n_pairs
    # Explicit symmetrization avoids floating-point asymmetry causing eigh failures.
    c0 = 0.5 * (c0 + c0.T)
    ctau = 0.5 * (ctau + ctau.T)
    c0_reg = c0 + regularization * np.eye(n_features)
    try:
        values, vectors = eigh(ctau, c0_reg, check_finite=True)
    except Exception as exc:
        raise ValueError("tICA generalized eigenproblem failed; check feature collinearity and regularization.") from exc
    order = np.argsort(np.abs(values))[::-1]
    values = values[order][:n_components]
    vectors = vectors[:, order][:, :n_components]
    # Normalize each eigenvector to unit variance under C0 metric.
    for i in range(vectors.shape[1]):
        norm2 = float(vectors[:, i].T @ c0_reg @ vectors[:, i])
        if norm2 > 0:
            vectors[:, i] /= np.sqrt(norm2)
    return TICAResult(mean, values, vectors.T, lag_time, c0, ctau)


def vamp2_score(trajectories: Sequence[np.ndarray], lag_time: int = 1, regularization: float = 1e-8) -> float:
    """Compute a finite-sample VAMP-2 score from lagged feature pairs.

    This descriptive score is useful for comparing feature sets or hyperparameters only
    when the same split strategy and data are used. For model selection, use trajectory-level
    cross-validation rather than scores computed on the training trajectories alone.
    """
    arrays, mean, n_pairs = _normalize_trajectories(trajectories, lag_time)
    p = arrays[0].shape[1]
    c00 = np.zeros((p, p)); c11 = np.zeros((p, p)); c01 = np.zeros((p, p))
    for a in arrays:
        x = a - mean
        x0, x1 = x[:-lag_time], x[lag_time:]
        c00 += x0.T @ x0
        c11 += x1.T @ x1
        c01 += x0.T @ x1
    c00 /= n_pairs; c11 /= n_pairs; c01 /= n_pairs

    def inv_sqrt(C: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(0.5 * (C + C.T))
        vals = np.maximum(vals, regularization)
        return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T

    K = inv_sqrt(c00) @ c01 @ inv_sqrt(c11)
    singular_values = np.linalg.svd(K, compute_uv=False)
    return float(np.sum(singular_values ** 2))


def implied_timescales(eigenvalues: np.ndarray, lag_time: int) -> np.ndarray:
    """Convert discrete-time eigenvalues to implied timescales in the same units as lag_time.

    Values with |lambda| >= 1 or lambda == 0 produce NaN because the usual reversible
    implied-timescale formula is not finite/meaningful for those estimates.
    """
    vals = np.abs(np.asarray(eigenvalues, dtype=float))
    if lag_time <= 0:
        raise ValueError("lag_time must be positive.")
    result = np.full(vals.shape, np.nan, dtype=float)
    valid = np.isfinite(vals) & (vals > 0) & (vals < 1)
    result[valid] = -float(lag_time) / np.log(vals[valid])
    return result
