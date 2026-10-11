"""Feature preparation, including explicit periodic-angle encoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureTransform:
    """Fitted feature transform metadata and scaler."""
    input_columns: list[str]
    output_columns: list[str]
    angle_degrees: list[str]
    angle_radians: list[str]
    scaler: StandardScaler | None

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.input_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing input feature columns: {missing}")
        out: list[np.ndarray] = []
        deg = set(self.angle_degrees)
        rad = set(self.angle_radians)
        for c in self.input_columns:
            x = pd.to_numeric(df[c], errors="raise").to_numpy(dtype=float)
            if c in deg:
                theta = np.deg2rad(x)
                out.extend([np.sin(theta), np.cos(theta)])
            elif c in rad:
                out.extend([np.sin(x), np.cos(x)])
            else:
                out.append(x)
        X = np.column_stack(out)
        if not np.isfinite(X).all():
            raise ValueError("Features contain NaN or infinite values; clean them before fitting.")
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X


def encode_features(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    angle_degrees: Iterable[str] = (),
    angle_radians: Iterable[str] = (),
    scale: bool = True,
) -> tuple[np.ndarray, FeatureTransform]:
    """Encode numeric features and optionally standardize them.

    Periodic angles are replaced by sin(theta), cos(theta) so values near -180/180 degrees
    or -pi/pi radians remain close in feature space. Distances and binary features remain
    separate scalar columns. This does not solve all mixed-feature weighting questions;
    see the methodological notes before combining heterogeneous observables.
    """
    columns = [str(c) for c in feature_columns]
    if not columns:
        raise ValueError("At least one feature column is required.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    deg = list(angle_degrees)
    rad = list(angle_radians)
    if set(deg) & set(rad):
        raise ValueError("A feature cannot be both degrees and radians.")
    for c in deg + rad:
        if c not in columns:
            raise ValueError(f"Angle feature {c!r} is not in feature_columns.")
    output_names: list[str] = []
    out: list[np.ndarray] = []
    for c in columns:
        x = pd.to_numeric(df[c], errors="raise").to_numpy(dtype=float)
        if c in deg:
            theta = np.deg2rad(x)
            out.extend([np.sin(theta), np.cos(theta)])
            output_names.extend([f"sin({c})", f"cos({c})"])
        elif c in rad:
            out.extend([np.sin(x), np.cos(x)])
            output_names.extend([f"sin({c})", f"cos({c})"])
        else:
            out.append(x)
            output_names.append(c)
    X = np.column_stack(out)
    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinite values; handle missing values explicitly.")
    scaler = StandardScaler() if scale else None
    if scaler is not None:
        X = scaler.fit_transform(X)
    transform = FeatureTransform(columns, output_names, deg, rad, scaler)
    return X, transform
