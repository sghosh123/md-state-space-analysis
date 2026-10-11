from pathlib import Path
from mdstate.cli import main


def test_demo_end_to_end(tmp_path):
    out = tmp_path / "demo"
    assert main(["demo", "--outdir", str(out), "--seed", "9"]) == 0
    for rel in [
        "synthetic_features.tsv",
        "results/pca_scores.tsv",
        "results/tica_scores.tsv",
        "results/clustered_features.tsv",
        "results/analysis_summary.json",
        "results/pca_colored_by_cluster.png",
    ]:
        assert (out / rel).is_file(), rel


def test_example_feature_spec_drives_cli(tmp_path):
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    spec = Path(__file__).parents[1] / "configs" / "yeast_mbd_chaperone_12d.json"
    cfg = json.loads(spec.read_text())
    rng = np.random.default_rng(3)
    rows = []
    for traj in ["T1", "T2"]:
        for frame in range(50):
            row = {"Trajectory": traj, "DCD Segment": 0, "Frame": frame}
            for c in cfg["feature_columns"]:
                row[c] = float(rng.normal(90, 20) if c in cfg["angle_degrees"] else rng.normal(20, 2))
            rows.append(row)
    input_path = tmp_path / "legacy_features.tsv"
    pd.DataFrame(rows).to_csv(input_path, sep="\t", index=False)
    output = tmp_path / "spec-results"
    assert main(["analyze", str(input_path), "--feature-spec", str(spec), "--outdir", str(output), "--lag-time", "3", "--pca-components", "2", "--tica-components", "2", "--n-clusters", "3"]) == 0
    assert (output / "feature_transform.json").is_file()
    assert (output / "tica_scores.tsv").is_file()
