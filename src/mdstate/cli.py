"""Command-line interface for reproducible feature-space analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from .clustering import cluster_summary, fit_dbscan, fit_hdbscan, fit_kmeans, score_clustering, kmeans_parameter_sweep, dbscan_parameter_sweep
from .diagnostics import evaluate_tica_lags
from .dimensionality import fit_pca, fit_tica
from .fes import estimate_fes, plot_fes_2d, write_fes_table
from .io import infer_feature_columns, read_feature_table, save_table, split_feature_trajectories
from .plotting import plot_explained_variance, plot_projection
from .preprocessing import encode_features


def _csv_list(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _resolve_feature_spec(args) -> None:
    """Fill unspecified feature/identifier choices from an optional JSON feature spec."""
    spec_path = getattr(args, "feature_spec", None)
    spec = {}
    if spec_path:
        spec = json.loads(Path(spec_path).read_text())
        if not isinstance(spec, dict):
            raise ValueError("Feature specification JSON must contain an object.")
        if getattr(args, "features", None) is None and spec.get("feature_columns"):
            args.features = ",".join(spec["feature_columns"])
        if getattr(args, "angle_degrees", None) is None and spec.get("angle_degrees"):
            args.angle_degrees = ",".join(spec["angle_degrees"])
        if getattr(args, "angle_radians", None) is None and spec.get("angle_radians"):
            args.angle_radians = ",".join(spec["angle_radians"])
        if getattr(args, "trajectory_column", None) is None:
            args.trajectory_column = spec.get("trajectory_column", "Trajectory")
        if getattr(args, "segment_column", None) is None:
            args.segment_column = spec.get("segment_column")
        if getattr(args, "frame_column", None) is None:
            args.frame_column = spec.get("frame_column", "Frame")
    else:
        if getattr(args, "trajectory_column", None) is None:
            args.trajectory_column = "Trajectory"
        if getattr(args, "frame_column", None) is None:
            args.frame_column = "Frame"


def _metadata_cols(df: pd.DataFrame, args) -> list[str]:
    return [c for c in (args.trajectory_column, args.segment_column, args.frame_column) if c and c in df.columns]


def _feature_matrix(df: pd.DataFrame, args):
    features = infer_feature_columns(df, _csv_list(args.features), metadata_columns=_metadata_cols(df, args))
    if not args.features:
        warnings.warn(f"Inferred {len(features)} numeric feature columns. For publication-quality analyses, pass --features explicitly.", RuntimeWarning)
    X, transform = encode_features(df, features, _csv_list(args.angle_degrees) or [], _csv_list(args.angle_radians) or [], scale=not args.no_scale)
    return features, X, transform


def _trajectory_groups(df, X, args):
    return split_feature_trajectories(X, df, args.trajectory_column, args.segment_column, args.frame_column)


def _write_metadata(df, args, outdir: Path) -> pd.DataFrame:
    cols = _metadata_cols(df, args)
    if cols:
        return df[cols].copy()
    return pd.DataFrame({"row_index": np.arange(len(df))})


def command_analyze(args) -> int:
    _resolve_feature_spec(args)
    df = read_feature_table(args.input, sep=args.sep)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    features, X, transform = _feature_matrix(df, args)
    if len(df) < 3:
        raise ValueError("At least three rows are needed for this combined analysis.")
    meta = _write_metadata(df, args, outdir)
    feature_mapping = []
    for c in transform.input_columns:
        if c in transform.angle_degrees or c in transform.angle_radians:
            feature_mapping.extend([{"input_feature": c, "encoded_feature": f"sin({c})"}, {"input_feature": c, "encoded_feature": f"cos({c})"}])
        else:
            feature_mapping.append({"input_feature": c, "encoded_feature": c})
    save_table(pd.DataFrame(feature_mapping), outdir / "feature_map.tsv")
    transform_json = {
        "input_columns": transform.input_columns,
        "output_columns": transform.output_columns,
        "angle_degrees": transform.angle_degrees,
        "angle_radians": transform.angle_radians,
        "scaled": transform.scaler is not None,
        "scaler_mean": transform.scaler.mean_.tolist() if transform.scaler is not None else None,
        "scaler_scale": transform.scaler.scale_.tolist() if transform.scaler is not None else None,
    }
    (outdir / "feature_transform.json").write_text(json.dumps(transform_json, indent=2) + "\n")

    # PCA
    pca_n = min(args.pca_components, X.shape[1], X.shape[0])
    pca, pca_scores = fit_pca(X, n_components=pca_n)
    pca_cols = [f"PC{i+1}" for i in range(pca_scores.shape[1])]
    pca_df = pd.concat([meta.reset_index(drop=True), pd.DataFrame(pca_scores, columns=pca_cols)], axis=1)
    save_table(pca_df, outdir / "pca_scores.tsv")
    save_table(pd.DataFrame({"component": pca_cols, "explained_variance_ratio": pca.explained_variance_ratio_, "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)}), outdir / "pca_explained_variance.tsv")
    if pca_scores.shape[1] >= 2:
        plot_projection(pca_scores, None, outdir / "pca_projection.png", "PCA projection (unclustered)")
    plot_explained_variance(pca, outdir / "pca_explained_variance.png")

    # tICA: rows remain tied to input positions; each trajectory/segment is fitted as a separate lag-pair sequence.
    groups = _trajectory_groups(df, X, args)
    tica = fit_tica(groups, lag_time=args.lag_time, n_components=min(args.tica_components, X.shape[1]), regularization=args.regularization)
    tic_parts = [tica.transform(g) for g in groups]
    # split_feature_trajectories optionally sorts within group by frame; place scores back in input order.
    index_cols = [c for c in (args.trajectory_column, args.segment_column) if c and c in df.columns]
    groups_frame = df
    if not index_cols:
        sorted_indices = [np.arange(len(df))]
    else:
        work = groups_frame[index_cols].copy(); work["__pos"] = np.arange(len(df))
        sorted_indices = []
        for _, group in work.groupby(index_cols, sort=False, dropna=False):
            pos = group["__pos"].to_numpy(dtype=int)
            if args.frame_column and args.frame_column in df.columns:
                frames = pd.to_numeric(df.iloc[pos][args.frame_column], errors="coerce").to_numpy()
                if np.isfinite(frames).all(): pos = pos[np.argsort(frames, kind="stable")]
            sorted_indices.append(pos)
    if len(sorted_indices) != len(tic_parts):
        raise RuntimeError("Internal trajectory grouping mismatch while writing tICA scores.")
    tics = np.full((len(df), tic_parts[0].shape[1]), np.nan, dtype=float)
    for pos, proj in zip(sorted_indices, tic_parts):
        tics[pos] = proj
    tic_cols = [f"TIC{i+1}" for i in range(tics.shape[1])]
    save_table(pd.concat([meta.reset_index(drop=True), pd.DataFrame(tics, columns=tic_cols)], axis=1), outdir / "tica_scores.tsv")
    tica_metrics = pd.DataFrame({"component": tic_cols, "eigenvalue": tica.eigenvalues_, "implied_timescale_frames": tica.implied_timescales})
    save_table(tica_metrics, outdir / "tica_eigenvalues.tsv")
    (outdir / "tica_metadata.json").write_text(json.dumps({"lag_time_frames": args.lag_time, "n_trajectory_segments": len(groups), "regularization": args.regularization, "warning": "Frame lag must be converted to physical time using the actual saved-frame interval."}, indent=2) + "\n")

    # Clustering in a chosen representation.
    if args.cluster_space == "features": cluster_X = X
    elif args.cluster_space == "pca": cluster_X = pca_scores
    else: cluster_X = tics
    if args.algorithm == "kmeans":
        _, labels = fit_kmeans(cluster_X, n_clusters=args.n_clusters, random_state=args.seed)
    elif args.algorithm == "dbscan":
        _, labels = fit_dbscan(cluster_X, eps=args.eps, min_samples=args.min_samples)
    else:
        _, labels = fit_hdbscan(cluster_X, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
    annotated = df.copy(); annotated["cluster_label"] = labels
    save_table(annotated, outdir / "clustered_features.tsv")
    save_table(cluster_summary(labels), outdir / "cluster_sizes.tsv")
    metrics = score_clustering(cluster_X, labels)
    metrics.update({"algorithm": args.algorithm, "cluster_space": args.cluster_space, "n_input_features": len(features), "n_encoded_features": int(X.shape[1]), "n_rows": len(df), "seed": args.seed})
    (outdir / "cluster_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
    if pca_scores.shape[1] >= 2:
        plot_projection(pca_scores, labels, outdir / "pca_colored_by_cluster.png", f"PCA colored by {args.algorithm} labels")
    summary = {
        "input": str(Path(args.input)), "n_rows": int(len(df)), "trajectory_segments": int(len(groups)),
        "original_features": features, "encoded_features": transform.output_columns,
        "algorithm": args.algorithm, "cluster_space": args.cluster_space,
        "pca_components": int(pca_n), "tica_components": int(tics.shape[1]), "lag_time_frames": int(args.lag_time),
        "note": "Clusters and tICA are exploratory until stability, lag-time and kinetic model validation are performed.",
    }
    (outdir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Analysis complete: {outdir.resolve()}")
    return 0


def command_lags(args) -> int:
    _resolve_feature_spec(args)
    if not getattr(args, "features", None):
        raise ValueError("Pass --features or --feature-spec; tICA lag diagnostics should not guess the feature set.")
    df = read_feature_table(args.input, sep=args.sep)
    features = infer_feature_columns(df, _csv_list(args.features), metadata_columns=[c for c in (args.trajectory_column, args.segment_column, args.frame_column) if c])
    X, transform = encode_features(df, features, _csv_list(args.angle_degrees) or [], _csv_list(args.angle_radians) or [], scale=not args.no_scale)
    groups = split_feature_trajectories(X, df, args.trajectory_column, args.segment_column, args.frame_column)
    result = evaluate_tica_lags(groups, [int(x) for x in args.lags.split(",") if x.strip()], n_components=min(args.components, X.shape[1]), regularization=args.regularization)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); save_table(result, out)
    print(f"tICA lag diagnostics written to {out.resolve()}")
    return 0


def command_cluster_sweep(args) -> int:
    _resolve_feature_spec(args)
    if not getattr(args, "features", None):
        raise ValueError("Pass --features or --feature-spec; clustering sweeps should not guess the feature set.")
    df = read_feature_table(args.input, sep=args.sep)
    features = infer_feature_columns(df, _csv_list(args.features), metadata_columns=[c for c in (args.trajectory_column, args.segment_column, args.frame_column) if c])
    X, _ = encode_features(df, features, _csv_list(args.angle_degrees) or [], _csv_list(args.angle_radians) or [], scale=not args.no_scale)
    if args.mode == "kmeans":
        result = kmeans_parameter_sweep(X, [int(v) for v in args.k_values.split(",") if v.strip()], [int(v) for v in args.seeds.split(",") if v.strip()])
    else:
        result = dbscan_parameter_sweep(X, [float(v) for v in args.eps_values.split(",") if v.strip()], [int(v) for v in args.min_samples_values.split(",") if v.strip()])
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); save_table(result, out)
    print(f"Parameter sweep written to {out.resolve()}")
    return 0


def command_fes(args) -> int:
    df = read_feature_table(args.input, sep=args.sep)
    columns = _csv_list(args.coordinates) or []
    if not columns:
        raise ValueError("Pass --coordinates with one, two, or three numeric coordinate columns.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Coordinate columns are missing: {missing}")
    coords = df[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    weights = pd.to_numeric(df[args.weights], errors="raise").to_numpy(dtype=float) if args.weights else None
    result = estimate_fes(coords, bins=[args.bins] * len(columns), temperature_kelvin=args.temperature, weights=weights)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    write_fes_table(result, outdir / "fes.tsv", coordinate_names=columns)
    if len(columns) == 2:
        plot_fes_2d(result, outdir / "fes_2d.png", coordinate_names=columns)
    metadata = {"coordinates": columns, "bins_per_dimension": args.bins, "temperature_kelvin": args.temperature, "kbt_kJ_mol": result.kbt_kj_mol, "weighted": weights is not None, "n_samples": int(len(coords)), "free_energy_convention": "-kBT ln(histogram probability mass), shifted so minimum occupied bin is zero", "warning": "For biased simulations, supply scientifically valid reweighting weights; this command does not compute them."}
    (outdir / "fes_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"FES written to {outdir.resolve()}")
    return 0


def command_demo(args) -> int:
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    names = [f"T{i}" for i in range(4)]
    n_frames = 400
    for traj in names:
        state = 0
        for frame in range(n_frames):
            if rng.random() < 0.025: state = 1 - state
            if state == 0:
                center = np.array([1.7, 2.0, 2.2, 2.4, 25., 45., 70., 110.])
            else:
                center = np.array([2.7, 2.5, 1.6, 1.8, 145., 120., 95., 40.])
            vals = center + np.array([0.12,0.15,0.1,0.18,8,8,7,10]) * rng.normal(size=8)
            rows.append({"Trajectory": traj, "Segment": 0, "Frame": frame,
                         "d_A1X1_nm": vals[0], "d_A1Y1_nm": vals[1], "d_A2X2_nm": vals[2], "d_B2Y2_nm": vals[3],
                         "angle_AX_deg": vals[4], "angle_BX_deg": vals[5], "angle_AY_deg": vals[6], "angle_BY_deg": vals[7]})
    input_path = outdir / "synthetic_features.tsv"
    pd.DataFrame(rows).to_csv(input_path, sep="\t", index=False)
    print(f"Wrote synthetic dataset: {input_path}")
    # construct analysis namespace matching parser options
    class A: pass
    a = A()
    for k, v in vars(args).items(): setattr(a, k, v)
    a.input = str(input_path); a.outdir = str(outdir / "results"); a.sep = "\t"
    a.features = "d_A1X1_nm,d_A1Y1_nm,d_A2X2_nm,d_B2Y2_nm,angle_AX_deg,angle_BX_deg,angle_AY_deg,angle_BY_deg"
    a.angle_degrees = "angle_AX_deg,angle_BX_deg,angle_AY_deg,angle_BY_deg"; a.angle_radians = None
    a.trajectory_column = "Trajectory"; a.segment_column = "Segment"; a.frame_column = "Frame"
    a.no_scale = False; a.pca_components = 3; a.tica_components = 3; a.lag_time = 5; a.regularization = 1e-8
    a.cluster_space = "features"; a.algorithm = "kmeans"; a.n_clusters = 2; a.seed = args.seed
    a.eps = 0.5; a.min_samples = 10; a.min_cluster_size = 20
    return command_analyze(a)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mdstate", description="Molecular-dynamics state-space analysis toolkit")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="Run PCA, tICA, clustering and basic diagnostics on a feature table")
    a.add_argument("input"); a.add_argument("--outdir", default="mdstate_results"); a.add_argument("--sep", default=None)
    a.add_argument("--features", help="Comma-separated columns to use; explicit columns are recommended"); a.add_argument("--feature-spec", default=None, help="JSON feature specification; explicit CLI values override it")
    a.add_argument("--angle-degrees", default=None); a.add_argument("--angle-radians", default=None)
    a.add_argument("--trajectory-column", default=None); a.add_argument("--segment-column", default=None); a.add_argument("--frame-column", default=None)
    a.add_argument("--no-scale", action="store_true"); a.add_argument("--pca-components", type=int, default=3); a.add_argument("--tica-components", type=int, default=3)
    a.add_argument("--lag-time", type=int, default=10, help="tICA lag in saved frames, not physical time")
    a.add_argument("--regularization", type=float, default=1e-8)
    a.add_argument("--algorithm", choices=["kmeans", "dbscan", "hdbscan"], default="kmeans")
    a.add_argument("--cluster-space", choices=["features", "pca", "tica"], default="features")
    a.add_argument("--n-clusters", type=int, default=5); a.add_argument("--eps", type=float, default=0.5); a.add_argument("--min-samples", type=int, default=10); a.add_argument("--min-cluster-size", type=int, default=20); a.add_argument("--seed", type=int, default=42)
    a.set_defaults(func=command_analyze)
    l = sub.add_parser("tica-lags", help="Evaluate tICA eigenvalues, timescales and VAMP2 over lag times")
    l.add_argument("input"); l.add_argument("--out", default="tica_lag_diagnostics.tsv"); l.add_argument("--sep", default=None)
    l.add_argument("--features", default=None); l.add_argument("--feature-spec", default=None, help="JSON feature specification"); l.add_argument("--angle-degrees", default=None); l.add_argument("--angle-radians", default=None)
    l.add_argument("--trajectory-column", default=None); l.add_argument("--segment-column", default=None); l.add_argument("--frame-column", default=None)
    l.add_argument("--no-scale", action="store_true"); l.add_argument("--lags", default="1,2,5,10,20"); l.add_argument("--components", type=int, default=3); l.add_argument("--regularization", type=float, default=1e-8)
    l.set_defaults(func=command_lags)
    cs = sub.add_parser("cluster-sweep", help="Evaluate clustering sensitivity over k/seeds or DBSCAN parameter grids")
    cs.add_argument("input"); cs.add_argument("--out", default="cluster_sweep.tsv"); cs.add_argument("--sep", default=None); cs.add_argument("--features", default=None); cs.add_argument("--feature-spec", default=None, help="JSON feature specification")
    cs.add_argument("--angle-degrees", default=None); cs.add_argument("--angle-radians", default=None); cs.add_argument("--no-scale", action="store_true")
    cs.add_argument("--trajectory-column", default=None); cs.add_argument("--segment-column", default=None); cs.add_argument("--frame-column", default=None)
    cs.add_argument("--mode", choices=["kmeans", "dbscan"], default="kmeans"); cs.add_argument("--k-values", default="2,3,4,5,6,7,8"); cs.add_argument("--seeds", default="42,43,44")
    cs.add_argument("--eps-values", default="0.2,0.3,0.4,0.5,0.7,1.0"); cs.add_argument("--min-samples-values", default="5,10,20")
    cs.set_defaults(func=command_cluster_sweep)
    f = sub.add_parser("fes", help="Estimate a 1D/2D/3D histogram free-energy surface from projected coordinates")
    f.add_argument("input"); f.add_argument("--coordinates", required=True, help="One to three coordinate columns, e.g. TIC1,TIC2")
    f.add_argument("--outdir", default="fes_results"); f.add_argument("--sep", default=None); f.add_argument("--bins", type=int, default=50)
    f.add_argument("--temperature", type=float, default=310.0, help="Temperature in kelvin; output FES is kJ/mol")
    f.add_argument("--weights", default=None, help="Optional column containing valid sample reweighting weights")
    f.set_defaults(func=command_fes)
    d = sub.add_parser("demo", help="Generate synthetic multi-trajectory data and run a complete demo")
    d.add_argument("--outdir", default="mdstate_demo"); d.add_argument("--seed", type=int, default=42); d.set_defaults(func=command_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, TypeError, FileNotFoundError, ImportError) as exc:
        parser.error(str(exc))
    return 2

if __name__ == "__main__":
    sys.exit(main())
