"""Join cluster assignments to mechanistic observables using explicit frame keys."""
from __future__ import annotations

import re
from typing import Sequence
import numpy as np
import pandas as pd

_OPERATORS = {"<", "<=", ">", ">=", "==", "!="}
_THRESHOLD_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_. -]*)\s*(<=|>=|==|!=|<|>)\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$")


def parse_threshold(expression: str) -> tuple[str, str, float]:
    """Parse e.g. ``CuS_distance_A<=3.5`` into (column, operator, value)."""
    match = _THRESHOLD_RE.match(expression)
    if not match:
        raise ValueError(
            f"Invalid threshold {expression!r}; use COLUMN<=VALUE, COLUMN>=VALUE, COLUMN<VALUE, COLUMN>VALUE, COLUMN==VALUE, or COLUMN!=VALUE."
        )
    column, operator, raw_value = match.groups()
    if operator not in _OPERATORS:
        raise ValueError(f"Unsupported threshold operator {operator!r}.")
    return column.strip(), operator, float(raw_value)


def _normalize_key_columns(table: pd.DataFrame, key_columns: Sequence[str], label: str) -> pd.DataFrame:
    result = table.copy()
    missing = [c for c in key_columns if c not in result.columns]
    if missing:
        raise ValueError(f"{label} is missing join-key columns: {missing}")
    if result[list(key_columns)].isna().any().any():
        raise ValueError(f"{label} contains missing join-key values.")
    for column in key_columns:
        # Explicit identity strings avoid accidental int-vs-string mismatches between TSVs.
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"{label} contains empty values in key column {column!r}.")
    if result.duplicated(list(key_columns)).any():
        example = result.loc[result.duplicated(list(key_columns), keep=False), list(key_columns)].head(3).to_dict("records")
        raise ValueError(f"{label} has duplicate rows for frame keys {list(key_columns)}; examples: {example}")
    return result


def _apply_threshold(values: pd.Series, operator: str, threshold: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if operator == "<": return numeric < threshold
    if operator == "<=": return numeric <= threshold
    if operator == ">": return numeric > threshold
    if operator == ">=": return numeric >= threshold
    if operator == "==": return numeric == threshold
    return numeric != threshold


def associate_clusters_with_mechanism(
    labels_table: pd.DataFrame,
    mechanism_table: pd.DataFrame,
    *,
    join_keys: Sequence[str] | None = None,
    label_keys: Sequence[str] | None = None,
    mechanism_keys: Sequence[str] | None = None,
    cluster_column: str = "cluster_label",
    trajectory_column: str = "trajectory_id",
    thresholds: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strictly join cluster labels and per-frame mechanistic observables.

    ``join_keys`` applies identical key-column names to both tables. Alternatively,
    ``label_keys`` and ``mechanism_keys`` can map equivalent identity fields with
    different names, in the same order (e.g. Trajectory/Segment/Frame to
    trajectory_id/segment_id/local_frame). Returns the joined frame table, cluster
    summary, and per-trajectory/cluster summary. Duplicate keys and mismatched key sets
    fail rather than permitting row-order joins or many-to-many expansion.
    """
    if labels_table.empty or mechanism_table.empty:
        raise ValueError("Label and mechanism tables must both contain at least one row.")
    if label_keys is None and mechanism_keys is None:
        if join_keys is None:
            raise ValueError("Pass join_keys or both label_keys and mechanism_keys.")
        label_cols = mechanism_cols = list(dict.fromkeys(str(k).strip() for k in join_keys if str(k).strip()))
    else:
        if label_keys is None or mechanism_keys is None:
            raise ValueError("Pass both label_keys and mechanism_keys when using differently named identity columns.")
        label_cols = list(dict.fromkeys(str(k).strip() for k in label_keys if str(k).strip()))
        mechanism_cols = list(dict.fromkeys(str(k).strip() for k in mechanism_keys if str(k).strip()))
        if len(label_cols) != len(mechanism_cols):
            raise ValueError("label_keys and mechanism_keys must have the same number of columns in matching order.")
    if not label_cols or not mechanism_cols:
        raise ValueError("Pass at least one explicit join key.")
    if len(label_cols) != len(set(label_cols)) or len(mechanism_cols) != len(set(mechanism_cols)):
        raise ValueError("Join key column names must be unique in each input.")
    if cluster_column in label_cols:
        raise ValueError("Cluster-label column cannot also be a frame identity join key.")
    if trajectory_column not in label_cols:
        raise ValueError(f"trajectory_column {trajectory_column!r} must be included in label_keys so replicate-aware summaries can be computed.")
    if cluster_column not in labels_table.columns:
        raise ValueError(f"Label table is missing cluster column {cluster_column!r}.")
    labels = labels_table[label_cols + [cluster_column]].copy()
    labels = _normalize_key_columns(labels, label_cols, "label table")
    mechanisms = mechanism_table.copy()
    mechanisms = _normalize_key_columns(mechanisms, mechanism_cols, "mechanism table")
    if labels[cluster_column].isna().any():
        raise ValueError("Cluster labels contain missing values.")

    canonical_keys = [f"__frame_key_{i}" for i in range(len(label_cols))]
    for canonical, label_key, mechanism_key in zip(canonical_keys, label_cols, mechanism_cols):
        labels[canonical] = labels[label_key]
        mechanisms[canonical] = mechanisms[mechanism_key]
    # Preserve the label table's identity column names in final output, and only carry
    # mechanism observables (not a second copy of differently named key fields).
    mechanism_value_columns = [c for c in mechanisms.columns if c not in mechanism_cols + canonical_keys]
    overlap = (set(labels.columns) & set(mechanism_value_columns)) - set(canonical_keys)
    overlap.discard(cluster_column)
    if overlap:
        raise ValueError(f"Mechanism table repeats label identity/cluster columns: {sorted(overlap)}")
    right = mechanisms[canonical_keys + mechanism_value_columns].copy()
    merged = labels.merge(right, on=canonical_keys, how="outer", validate="one_to_one", indicator=True, sort=False)
    if not merged["_merge"].eq("both").all():
        counts = merged["_merge"].value_counts().to_dict()
        examples = merged.loc[merged["_merge"] != "both", label_cols + ["_merge"]].head(5).to_dict("records")
        raise ValueError(f"Label/mechanism frame keys do not match exactly (counts={counts}); examples: {examples}")
    merged = merged.drop(columns=canonical_keys + ["_merge"])

    event_columns = []
    event_names: dict[str, str] = {}
    for expression in thresholds:
        column, operator, threshold = parse_threshold(expression)
        if column not in merged.columns:
            raise ValueError(f"Threshold column {column!r} is absent from joined data.")
        if column in label_cols or column in mechanism_cols:
            raise ValueError(f"Threshold column {column!r} cannot be a join key.")
        safe_operator = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}[operator]
        safe_value = format(threshold, ".8g").replace("-", "m").replace(".", "p")
        event_name = f"event__{column.replace(' ', '_')}__{safe_operator}_{safe_value}"
        if event_name in merged.columns or event_name in event_columns:
            raise ValueError(f"Repeated event threshold creates duplicate output name: {event_name}")
        merged[event_name] = _apply_threshold(merged[column], operator, threshold).astype("boolean")
        event_columns.append(event_name)
        event_names[event_name] = expression

    # Numeric mechanism observables exclude identifiers, labels, and generated booleans.
    metric_cols = [c for c in mechanism_value_columns if pd.api.types.is_numeric_dtype(mechanisms[c])]
    if not metric_cols and not event_columns:
        raise ValueError("Mechanism table has no numeric observables or requested event thresholds.")
    # For summaries, coerce numeric metrics explicitly but reject columns that contain
    # malformed strings rather than producing misleading partial statistics.
    for column in metric_cols:
        converted = pd.to_numeric(merged[column], errors="coerce")
        if merged[column].notna().sum() != converted.notna().sum():
            raise ValueError(f"Mechanism observable {column!r} contains non-numeric values.")
        merged[column] = converted

    group_keys = [cluster_column]
    cluster_rows = []
    for cluster, group in merged.groupby(cluster_column, sort=True, dropna=False):
        record = {cluster_column: cluster, "n_frames": int(len(group)),
                  "n_trajectories": int(group[trajectory_column].nunique())}
        for column in metric_cols:
            record[f"{column}__mean"] = float(group[column].mean()) if group[column].notna().any() else float("nan")
            record[f"{column}__median"] = float(group[column].median()) if group[column].notna().any() else float("nan")
            record[f"{column}__std_frames"] = float(group[column].std(ddof=1)) if group[column].notna().sum() > 1 else float("nan")
        for event in event_columns:
            record[f"{event}__frame_fraction"] = float(group[event].mean()) if group[event].notna().any() else float("nan")
        cluster_rows.append(record)
    cluster_summary = pd.DataFrame(cluster_rows)

    # Per-trajectory means/fractions expose heterogeneity and prevent treating frames as
    # independent replicates when the user performs across-run statistical comparisons.
    traj_rows = []
    for (trajectory, cluster), group in merged.groupby([trajectory_column, cluster_column], sort=True, dropna=False):
        record = {trajectory_column: trajectory, cluster_column: cluster, "n_frames": int(len(group))}
        for column in metric_cols:
            record[f"{column}__mean"] = float(group[column].mean()) if group[column].notna().any() else float("nan")
        for event in event_columns:
            record[f"{event}__fraction"] = float(group[event].mean()) if group[event].notna().any() else float("nan")
        traj_rows.append(record)
    per_trajectory = pd.DataFrame(traj_rows)
    merged.attrs["event_expressions"] = event_names
    return merged, cluster_summary, per_trajectory
