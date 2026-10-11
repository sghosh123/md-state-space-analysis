"""Histogram-based free-energy surfaces from projected or collective-variable samples."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

GAS_CONSTANT_KJ_MOL_K = 0.00831446261815324


@dataclass
class FESResult:
    edges: list[np.ndarray]
    probability: np.ndarray
    free_energy: np.ndarray
    kbt_kj_mol: float

    @property
    def centers(self) -> list[np.ndarray]:
        return [(e[:-1] + e[1:]) * 0.5 for e in self.edges]


def estimate_fes(
    coordinates: np.ndarray,
    bins: int | Sequence[int] = 50,
    temperature_kelvin: float = 310.0,
    weights: np.ndarray | None = None,
    ranges: Sequence[tuple[float, float]] | None = None,
) -> FESResult:
    """Estimate a histogram free-energy surface, with its minimum shifted to zero.

    Samples should represent an equilibrium, unbiased distribution or have appropriate
    statistical weights supplied (for example, valid reweighting weights from a biased
    simulation). This function does not infer or generate metadynamics reweighting weights.
    Empty bins have infinite free energy; no pseudocount is added silently.

    Parameters
    ----------
    coordinates:
        Array shaped ``(n_samples, n_dimensions)`` for 1D, 2D, or 3D surfaces.
    bins:
        Number of bins per dimension or one count per dimension.
    temperature_kelvin:
        Temperature used for the Boltzmann factor. Output free energy uses kJ/mol.
    weights:
        Optional finite non-negative sample weights, one per sample.
    ranges:
        Optional histogram bounds per dimension.
    """
    X = np.asarray(coordinates, dtype=float)
    if X.ndim != 2 or X.shape[0] < 1 or not 1 <= X.shape[1] <= 3:
        raise ValueError("coordinates must be (n_samples, n_dimensions) for 1D, 2D or 3D FES.")
    if not np.isfinite(X).all():
        raise ValueError("coordinates contain NaN or infinite values.")
    if not np.isfinite(temperature_kelvin) or temperature_kelvin <= 0:
        raise ValueError("temperature_kelvin must be positive and finite.")
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or len(w) != len(X):
            raise ValueError("weights must have one value per coordinate row.")
        if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be finite, non-negative, and have positive sum.")
    else:
        w = None
    hist, edges = np.histogramdd(X, bins=bins, range=ranges, weights=w)
    total = float(hist.sum())
    if total <= 0:
        raise ValueError("No coordinate samples fell inside the requested histogram range.")
    probability = hist / total
    kbt = GAS_CONSTANT_KJ_MOL_K * float(temperature_kelvin)
    fes = np.full(probability.shape, np.inf, dtype=float)
    occupied = probability > 0
    fes[occupied] = -kbt * np.log(probability[occupied])
    fes[occupied] -= np.min(fes[occupied])
    return FESResult([np.asarray(e) for e in edges], probability, fes, kbt)


def write_fes_table(result: FESResult, path: str | Path, coordinate_names: Sequence[str] | None = None) -> None:
    """Write bin centers, probability masses and free energies as TSV."""
    import pandas as pd
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    names = list(coordinate_names or [f"CV{i+1}" for i in range(result.probability.ndim)])
    if len(names) != result.probability.ndim:
        raise ValueError("coordinate_names length must match FES dimensionality.")
    grids = np.meshgrid(*result.centers, indexing="ij")
    data = {name: grid.ravel() for name, grid in zip(names, grids)}
    data["probability_mass"] = result.probability.ravel()
    data["free_energy_kJ_mol"] = result.free_energy.ravel()
    pd.DataFrame(data).to_csv(path, sep="\t", index=False)


def plot_fes_2d(result: FESResult, path: str | Path, coordinate_names: Sequence[str] | None = None) -> None:
    """Plot a 2D free-energy surface, leaving unsampled bins masked."""
    import matplotlib.pyplot as plt
    if result.probability.ndim != 2:
        raise ValueError("plot_fes_2d requires a two-dimensional FES.")
    names = list(coordinate_names or ["CV1", "CV2"])
    if len(names) != 2:
        raise ValueError("Exactly two coordinate names are required.")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.8))
    plotted = np.ma.masked_invalid(np.where(np.isfinite(result.free_energy), result.free_energy, np.nan))
    m = ax.pcolormesh(result.edges[0], result.edges[1], plotted.T, shading="auto")
    fig.colorbar(m, ax=ax, label="Free energy (kJ/mol)")
    ax.set_xlabel(names[0]); ax.set_ylabel(names[1]); ax.set_title("Histogram free-energy surface")
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)
