import numpy as np
import pytest
from mdstate.fes import estimate_fes, write_fes_table, plot_fes_2d


def test_fes_normalizes_probability_and_minimum_zero(tmp_path):
    rng = np.random.default_rng(12)
    X = np.vstack([rng.normal(-1, 0.2, (500, 2)), rng.normal(1, 0.2, (500, 2))])
    result = estimate_fes(X, bins=25, temperature_kelvin=310)
    assert np.isclose(result.probability.sum(), 1.0)
    assert np.isclose(np.min(result.free_energy[np.isfinite(result.free_energy)]), 0.0)
    table = tmp_path / "fes.tsv"
    write_fes_table(result, table, ["TIC1", "TIC2"])
    assert table.exists()
    png = tmp_path / "fes.png"
    plot_fes_2d(result, png, ["TIC1", "TIC2"])
    assert png.exists()


def test_fes_weights_and_bad_input():
    X = np.array([[0.], [1.], [2.]])
    r = estimate_fes(X, bins=3, weights=np.array([1., 5., 1.]))
    assert r.probability.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive"):
        estimate_fes(X, weights=np.zeros(3))
