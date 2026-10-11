import numpy as np
import pandas as pd
import pytest
from mdstate.io import infer_feature_columns, split_feature_trajectories, read_feature_table


def test_infer_excludes_metadata_and_labels():
    df = pd.DataFrame({"Trajectory": ["T1", "T1"], "Frame": [0, 1], "cluster_id": [1, 1], "dist": [1.0, 2.0]})
    assert infer_feature_columns(df) == ["dist"]


def test_explicit_features_validate():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="missing"):
        infer_feature_columns(df, ["oops"])


def test_splits_trajectory_and_segment_and_sorts_frames():
    df = pd.DataFrame({"Trajectory": ["T1", "T1", "T1", "T2"], "Segment": [0, 1, 0, 0], "Frame": [1, 0, 0, 0]})
    X = np.array([[10.], [20.], [30.], [40.]])
    groups = split_feature_trajectories(X, df, "Trajectory", "Segment", "Frame")
    assert len(groups) == 3
    assert groups[0].ravel().tolist() == [30., 10.]
    assert groups[1].ravel().tolist() == [20.]
    assert groups[2].ravel().tolist() == [40.]


def test_read_table_duplicate_columns(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("x,x\n1,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_feature_table(p, sep=",")
