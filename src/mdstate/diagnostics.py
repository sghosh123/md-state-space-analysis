"""Lag-time and clustering diagnostics."""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd

from .dimensionality import fit_tica, vamp2_score


def evaluate_tica_lags(
    trajectories: Sequence[np.ndarray],
    lag_times: Sequence[int],
    n_components: int = 3,
    regularization: float = 1e-8,
) -> pd.DataFrame:
    """Fit tICA at each requested frame lag and report eigenvalues/timescales/VAMP2.

    Rows from too-short trajectory segments are handled by the core tICA estimator. This
    function does not claim convergence merely because curves look flat; it generates
    diagnostics for interpretation.
    """
    if not lag_times:
        raise ValueError("At least one lag time is required.")
    rows = []
    for lag in lag_times:
        result = fit_tica(trajectories, lag_time=int(lag), n_components=n_components, regularization=regularization)
        score = vamp2_score(trajectories, lag_time=int(lag), regularization=regularization)
        for i, (eig, ts) in enumerate(zip(result.eigenvalues_, result.implied_timescales), start=1):
            rows.append({"lag_time_frames": int(lag), "component": i, "eigenvalue": float(eig), "implied_timescale_frames": float(ts), "vamp2_score": score})
    return pd.DataFrame(rows)
