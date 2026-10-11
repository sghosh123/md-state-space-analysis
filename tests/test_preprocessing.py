import numpy as np
import pandas as pd
import pytest
from mdstate.preprocessing import encode_features


def test_periodic_angles_map_wraparound_neighbors_close():
    df = pd.DataFrame({"angle": [179.0, -179.0], "distance": [1.0, 1.0]})
    X, spec = encode_features(df, ["angle", "distance"], angle_degrees=["angle"], scale=False)
    assert X.shape == (2, 3)
    assert np.linalg.norm(X[0, :2] - X[1, :2]) < 0.04
    assert spec.output_columns == ["sin(angle)", "cos(angle)", "distance"]


def test_rejects_nan():
    with pytest.raises(ValueError, match="NaN or infinite"):
        encode_features(pd.DataFrame({"x": [1.0, np.nan]}), ["x"])


def test_rejects_angle_not_in_features():
    with pytest.raises(ValueError, match="not in feature_columns"):
        encode_features(pd.DataFrame({"x": [1, 2]}), ["x"], angle_degrees=["theta"])
