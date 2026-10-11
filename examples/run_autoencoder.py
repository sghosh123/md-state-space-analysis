"""Optional example: train an autoencoder after splitting data by trajectory.

This example intentionally uses synthetic data. For real MD, encode/scaling parameters must
be fit on training trajectories only and then applied to held-out trajectories.
"""
from __future__ import annotations
import numpy as np
from mdstate.autoencoder import fit_autoencoder

rng = np.random.default_rng(19)
latent = rng.normal(size=(1200, 3))
projection = rng.normal(size=(3, 10))
X = latent @ projection + 0.05 * rng.normal(size=(1200, 10))
# In a real simulation, train/validation arrays should come from different trajectories/replicas.
result = fit_autoencoder(X[:900], X_validation=X[900:], latent_dim=2, epochs=80, seed=19)
print("best epoch:", result.config["best_epoch"])
print("latent shape:", result.encode(X[900:]).shape)
print("validation reconstruction MSE:", float(np.mean((result.reconstruct(X[900:]) - X[900:]) ** 2)))
result.save("autoencoder_demo.pt")
