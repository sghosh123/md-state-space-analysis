import pandas as pd
import pytest

from mdstate.association import associate_clusters_with_mechanism, parse_threshold


def test_parse_threshold():
    assert parse_threshold("CuS_distance_A <= 3.5") == ("CuS_distance_A", "<=", 3.5)
    with pytest.raises(ValueError):
        parse_threshold("distance around 3")


def test_association_joins_by_keys_not_row_order_and_reports_replicates():
    labels = pd.DataFrame({
        "Trajectory": ["T1", "T1", "T1", "T2", "T2", "T2"],
        "Segment": [0, 0, 0, 0, 0, 0], "Frame": [0, 1, 2, 0, 1, 2],
        "cluster_label": [0, 0, 1, 0, 1, 1],
    })
    mechanism = pd.DataFrame({
        "Trajectory": ["T2", "T1", "T2", "T1", "T2", "T1"],
        "Segment": [0] * 6, "Frame": [2, 0, 0, 2, 1, 1],
        "CuS_distance_A": [2.5, 4.0, 4.5, 2.0, 3.0, 2.5],
        "salt_bridge_count": [1, 0, 0, 2, 1, 1],
    })
    joined, summary, by_traj = associate_clusters_with_mechanism(
        labels, mechanism, join_keys=["Trajectory", "Segment", "Frame"],
        cluster_column="cluster_label", trajectory_column="Trajectory",
        thresholds=["CuS_distance_A<=3.5", "salt_bridge_count>=1"],
    )
    assert len(joined) == 6
    assert joined.loc[(joined.Trajectory == "T1") & (joined.Frame == "0"), "CuS_distance_A"].item() == 4.0
    assert set(by_traj["Trajectory"]) == {"T1", "T2"}
    assert summary["cluster_label"].nunique() == 2
    assert any(col.endswith("__frame_fraction") for col in summary.columns)


def test_association_rejects_duplicate_frame_keys():
    labels = pd.DataFrame({"trajectory_id": ["T1", "T1"], "frame": [0, 0], "cluster_label": [0, 1]})
    mechanism = pd.DataFrame({"trajectory_id": ["T1"], "frame": [0], "CuS_distance_A": [2.0]})
    with pytest.raises(ValueError, match="duplicate rows"):
        associate_clusters_with_mechanism(labels, mechanism, join_keys=["trajectory_id", "frame"])


def test_association_rejects_mismatched_frame_sets():
    labels = pd.DataFrame({"trajectory_id": ["T1", "T1"], "frame": [0, 1], "cluster_label": [0, 1]})
    mechanism = pd.DataFrame({"trajectory_id": ["T1", "T1"], "frame": [0, 2], "CuS_distance_A": [2.0, 3.0]})
    with pytest.raises(ValueError, match="do not match exactly"):
        associate_clusters_with_mechanism(labels, mechanism, join_keys=["trajectory_id", "frame"])


def test_associate_cli_writes_frame_and_replicate_summaries(tmp_path):
    from mdstate.cli import main
    labels = pd.DataFrame({"Trajectory": ["T1", "T1", "T2", "T2"], "Segment": [0, 0, 0, 0],
                           "Frame": [0, 1, 0, 1], "cluster_label": [0, 1, 0, 1]})
    mechanism = pd.DataFrame({"Trajectory": ["T2", "T1", "T2", "T1"], "Segment": [0, 0, 0, 0],
                              "Frame": [1, 0, 0, 1], "CuS_distance_A": [2.8, 4.2, 3.8, 2.9]})
    labels_path = tmp_path / "labels.tsv"
    mech_path = tmp_path / "mechanism.tsv"
    labels.to_csv(labels_path, sep="\t", index=False)
    mechanism.to_csv(mech_path, sep="\t", index=False)
    outdir = tmp_path / "results"
    assert main(["associate", str(labels_path), str(mech_path), "--join-keys", "Trajectory,Segment,Frame",
                 "--trajectory-column", "Trajectory", "--threshold", "CuS_distance_A<=3.5",
                 "--outdir", str(outdir)]) == 0
    assert (outdir / "cluster_mechanism_frames.tsv").exists()
    summary = pd.read_csv(outdir / "cluster_mechanism_summary.tsv", sep="\t")
    assert len(summary) == 2
    assert (outdir / "cluster_mechanism_by_trajectory.tsv").exists()


def test_association_maps_different_key_names_from_analysis_repositories():
    labels = pd.DataFrame({
        "Trajectory": ["T1", "T1", "T2", "T2"],
        "Segment": [0, 0, 0, 0],
        "Frame": [0, 1, 0, 1],
        "cluster_label": [0, 1, 0, 1],
    })
    mechanism = pd.DataFrame({
        "trajectory_id": ["T2", "T1", "T2", "T1"],
        "segment_id": [0, 0, 0, 0],
        "local_frame": [1, 0, 0, 1],
        "CuS_distance_A": [2.8, 4.2, 3.8, 2.9],
        "salt_bridge_count": [1, 0, 2, 1],
    })
    joined, summary, by_trajectory = associate_clusters_with_mechanism(
        labels, mechanism,
        label_keys=["Trajectory", "Segment", "Frame"],
        mechanism_keys=["trajectory_id", "segment_id", "local_frame"],
        trajectory_column="Trajectory", cluster_column="cluster_label",
        thresholds=["CuS_distance_A<=3.5", "salt_bridge_count>=1"],
    )
    assert len(joined) == 4
    row = joined.loc[(joined.Trajectory == "T1") & (joined.Frame == "0")].iloc[0]
    assert row["CuS_distance_A"] == 4.2
    assert len(summary) == 2
    assert set(by_trajectory["Trajectory"]) == {"T1", "T2"}
    assert joined.filter(like="event__").shape[1] == 2
