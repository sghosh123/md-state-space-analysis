"""Basic kinetic diagnostics from preassigned discrete states.

These helpers are diagnostics, not a substitute for model validation or a specialized
MSM framework. In particular, the simple transition estimator here should not be used to
make kinetic claims until implied-timescale and Chapman–Kolmogorov tests are considered.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

from .dimensionality import implied_timescales


def transition_matrix(
    state_trajectories: Sequence[np.ndarray],
    lag_time: int = 1,
    reversible_counts: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a row-normalized transition matrix without crossing trajectory boundaries.

    Returns ``(transition_matrix, state_labels)``. Noise/unassigned state ``-1`` is ignored.
    State labels are sorted unique labels, preserving the mapping to input assignments.
    """
    if not isinstance(lag_time, int) or lag_time < 1:
        raise ValueError("lag_time must be a positive integer number of frames.")
    arrays = [np.asarray(t, dtype=int) for t in state_trajectories]
    if not arrays or any(a.ndim != 1 for a in arrays):
        raise ValueError("Pass one or more one-dimensional state-label trajectories.")
    states = np.unique(np.concatenate(arrays))
    states = states[states != -1]
    if len(states) < 2:
        raise ValueError("At least two non-noise states are needed to estimate transitions.")
    index = {int(s): i for i, s in enumerate(states)}
    counts = np.zeros((len(states), len(states)), dtype=float)
    for arr in arrays:
        if len(arr) <= lag_time:
            continue
        for a, b in zip(arr[:-lag_time], arr[lag_time:]):
            if a == -1 or b == -1:
                continue
            counts[index[int(a)], index[int(b)]] += 1
    if reversible_counts:
        counts = 0.5 * (counts + counts.T)
    row_sums = counts.sum(axis=1)
    T = np.zeros_like(counts)
    nonzero = row_sums > 0
    T[nonzero] = counts[nonzero] / row_sums[nonzero, None]
    # A state with no outgoing observed transitions is not a well-defined Markov row.
    # It is kept as all-zero and should be interpreted as unsupported, not as absorbing.
    return T, states


def transition_implied_timescales(T: np.ndarray, lag_time: int) -> pd.DataFrame:
    """Return implied timescales from a transition matrix's eigenvalues.

    The stationary eigenvalue nearest 1 is dropped. These estimates are only diagnostic
    until state discretization, sampling, lag dependence, and model validation are checked.
    """
    T = np.asarray(T, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1] or T.shape[0] < 2:
        raise ValueError("T must be a square transition matrix with at least two states.")
    if not np.isfinite(T).all() or (T < 0).any():
        raise ValueError("Transition matrix must contain finite non-negative values.")
    row_sums = T.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("Transition matrix rows must sum to one; unsupported zero rows must be resolved first.")
    eigvals = np.linalg.eigvals(T)
    eigvals = eigvals[np.argsort(np.abs(eigvals))[::-1]]
    if len(eigvals):
        eigvals = eigvals[1:]  # remove eigenvalue closest to stationary mode
    times = implied_timescales(eigvals, lag_time)
    return pd.DataFrame({"eigenvalue_real": np.real(eigvals), "eigenvalue_imag": np.imag(eigvals), "implied_timescale_frames": times})
