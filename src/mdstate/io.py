"""Validated table loading and trajectory-boundary handling."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import csv
import re

import numpy as np
import pandas as pd

DEFAULT_METADATA = {
    "trajectory", "traj", "frame", "segment", "dcd segment", "time", "time_ps", "time_ns",
    "cluster", "cluster_id", "cluster_label", "label", "state", "state_id", "basin_id",
    "microstate", "macrostate", "index", "row_id",
}
_LABEL_RE = re.compile(r"(^|[_ -])(cluster|label|state|microstate|macrostate|basin)([_ -]|$)", re.I)


def read_feature_table(path: str | Path, sep: str | None = None) -> pd.DataFrame:
    """Read a delimited feature table, validate column names and preserve row order.

    Tab-separated files are common in the associated research workflow. If ``sep`` is
    omitted, pandas' delimiter inference is used. Empty tables and duplicate columns fail
    loudly rather than flowing into downstream estimators.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input table does not exist: {path}")
    # pandas silently renames duplicate headers (for example x, x -> x, x.1), so inspect
    # the raw header first to fail explicitly instead of masking a common data issue.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        if sep is None:
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters="\t,;|")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","
        else:
            delimiter = sep
        raw_header = next(csv.reader(handle, delimiter=delimiter), [])
    stripped_header = [h.strip() for h in raw_header]
    if len(stripped_header) != len(set(stripped_header)):
        raise ValueError("Input table has duplicate column names; make them unique before analysis.")
    if sep is None:
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        df = pd.read_csv(path, sep=sep)
    if df.empty:
        raise ValueError(f"Input table contains no rows: {path}")
    if any(not str(c).strip() for c in df.columns):
        raise ValueError("Input table has an empty column name.")
    return df


def infer_feature_columns(
    df: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
    metadata_columns: Iterable[str] = (),
) -> list[str]:
    """Resolve numeric feature columns while excluding identifiers and existing labels.

    Explicit ``feature_columns`` are preferred in production work. Automatic inference is
    intended as a convenience for clean feature tables and deliberately excludes common
    metadata/label fields.
    """
    if feature_columns is not None:
        cols = [str(c).strip() for c in feature_columns if str(c).strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns are missing: {missing}")
        if len(cols) != len(set(cols)):
            raise ValueError("Feature column list contains duplicates.")
    else:
        meta = {str(c).strip().lower() for c in metadata_columns} | DEFAULT_METADATA
        cols = []
        for c in df.columns:
            low = str(c).strip().lower()
            if low in meta or _LABEL_RE.search(low):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    if not cols:
        raise ValueError("No feature columns selected. Pass --features with the intended numeric columns.")
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise TypeError(f"Feature column {c!r} is not numeric.")
    return cols


def split_feature_trajectories(
    X: np.ndarray,
    df: pd.DataFrame,
    trajectory_column: str | None = "Trajectory",
    segment_column: str | None = None,
    frame_column: str | None = "Frame",
) -> list[np.ndarray]:
    """Split features into independent continuous trajectories/segments.

    Lagged pairs are never allowed to cross a trajectory or segment boundary. If the
    relevant columns are absent, the full array is treated as one trajectory and a warning
    is not automatically emitted; callers should pass identifiers whenever available.
    Within each group, rows are stably ordered by ``frame_column`` when it exists.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) != len(df):
        raise ValueError("X must be a 2D array with one row per dataframe row.")
    group_cols: list[str] = []
    for c in (trajectory_column, segment_column):
        if c and c in df.columns and c not in group_cols:
            group_cols.append(c)
    if not group_cols:
        return [X.copy()]
    work = df[group_cols].copy()
    work["__row_position__"] = np.arange(len(df))
    groups: list[np.ndarray] = []
    # dropna=False keeps a metadata-missing group visible rather than dropping data silently.
    for _, g in work.groupby(group_cols, sort=False, dropna=False):
        positions = g["__row_position__"].to_numpy(dtype=int)
        if frame_column and frame_column in df.columns:
            frames = pd.to_numeric(df.iloc[positions][frame_column], errors="coerce").to_numpy()
            if np.isfinite(frames).all():
                positions = positions[np.argsort(frames, kind="stable")]
        groups.append(X[positions].copy())
    return groups


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a TSV file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
