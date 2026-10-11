import numpy as np
import pytest

torch = pytest.importorskip("torch")
from mdstate.autoencoder import fit_autoencoder


def test_autoencoder_fit_encode_reconstruct(tmp_path):
    rng = np.random.default_rng(4)
    latent = rng.normal(size=(120, 2))
    X = latent @ rng.normal(size=(2, 5)) + 0.03 * rng.normal(size=(120, 5))
    result = fit_autoencoder(X[:90], X_validation=X[90:], latent_dim=2, hidden_dims=(8,), epochs=8, batch_size=16, patience=3, seed=4)
    z = result.encode(X[90:])
    rec = result.reconstruct(X[90:])
    assert z.shape == (30, 2)
    assert rec.shape == X[90:].shape
    p = tmp_path / "ae.pt"
    result.save(p)
    assert p.exists()
    assert result.history
