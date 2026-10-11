"""Optional PyTorch autoencoder for latent-space exploration.

This is a new generic baseline; no autoencoder source script was present in the supplied ZIP
segments. Input should already be feature-encoded and scaled using a documented transform.
For validation, pass held-out trajectories via ``X_validation`` rather than randomly splitting
correlated MD frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import copy
import numpy as np


@dataclass
class AutoencoderResult:
    model: object
    history: list[dict[str, float]]
    input_dim: int
    latent_dim: int
    config: dict

    def encode(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        torch = _torch()
        X = _matrix(X, self.input_dim)
        device = next(self.model.parameters()).device
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                tensor = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
                z = self.model.encoder(tensor).cpu().numpy()
                outputs.append(z)
        return np.concatenate(outputs, axis=0)

    def reconstruct(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        torch = _torch()
        X = _matrix(X, self.input_dim)
        device = next(self.model.parameters()).device
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                tensor = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
                outputs.append(self.model(tensor).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def save(self, path: str | Path) -> None:
        torch = _torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "config": self.config, "history": self.history}, path)


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Autoencoder support requires `pip install 'md-state-space-analysis[latent]'`.") from exc
    return torch


def _matrix(X: np.ndarray, n_features: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or len(X) < 1:
        raise ValueError("X must be a non-empty two-dimensional array.")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Input feature dimension differs from the fitted autoencoder.")
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinite values.")
    return X


def fit_autoencoder(
    X_train: np.ndarray,
    X_validation: np.ndarray | None = None,
    latent_dim: int = 2,
    hidden_dims: Sequence[int] = (64, 32),
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    patience: int = 15,
    seed: int = 42,
    device: str = "cpu",
) -> AutoencoderResult:
    """Train a dense autoencoder with optional explicit validation data and early stopping."""
    torch = _torch()
    X = _matrix(X_train)
    Xv = _matrix(X_validation, X.shape[1]) if X_validation is not None else None
    if latent_dim < 1 or latent_dim >= X.shape[1]:
        raise ValueError("latent_dim must be >=1 and smaller than input feature count.")
    if not hidden_dims or any(int(h) < 1 for h in hidden_dims):
        raise ValueError("hidden_dims must contain positive integers.")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or patience < 1:
        raise ValueError("epochs, batch_size, patience and learning_rate must be positive.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    widths = [X.shape[1], *[int(h) for h in hidden_dims], int(latent_dim)]
    enc = []
    for i in range(len(widths) - 1):
        enc.append(torch.nn.Linear(widths[i], widths[i + 1]))
        if i < len(widths) - 2:
            enc.append(torch.nn.ReLU())
    decoder_widths = [int(latent_dim), *[int(h) for h in reversed(hidden_dims)], X.shape[1]]
    dec = []
    for i in range(len(decoder_widths) - 1):
        dec.append(torch.nn.Linear(decoder_widths[i], decoder_widths[i + 1]))
        if i < len(decoder_widths) - 2:
            dec.append(torch.nn.ReLU())
    class _AE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(*enc)
            self.decoder = torch.nn.Sequential(*dec)
        def forward(self, data):
            return self.decoder(self.encoder(data))
    model = _AE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    train_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    val_t = torch.as_tensor(Xv, dtype=torch.float32, device=device) if Xv is not None else None
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_t), generator=generator).to(device)
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = train_t[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(idx)
        train_loss = total / len(train_t)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_t), val_t).cpu()) if val_t is not None else train_loss
        history.append({"epoch": float(epoch), "train_mse": train_loss, "validation_mse": val_loss})
        if val_loss < best_loss - 1e-9:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if val_t is not None and stale >= patience:
                break
    model.load_state_dict(best_state)
    config = {"input_dim": int(X.shape[1]), "latent_dim": int(latent_dim), "hidden_dims": [int(h) for h in hidden_dims], "epochs_requested": int(epochs), "best_epoch": int(best_epoch), "learning_rate": float(learning_rate), "seed": int(seed), "device": str(device), "validation_used": Xv is not None}
    return AutoencoderResult(model, history, X.shape[1], latent_dim, config)
