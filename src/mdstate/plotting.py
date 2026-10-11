"""Plot helpers with explicit labels and no implicit scientific claims."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_projection(scores: np.ndarray, labels: np.ndarray | None, path: str | Path, title: str = "PCA projection") -> None:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("At least two projection components are required for a scatter plot.")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    if labels is None:
        ax.scatter(scores[:, 0], scores[:, 1], s=10, alpha=0.65)
    else:
        labels = np.asarray(labels)
        if len(labels) != len(scores):
            raise ValueError("labels must have one value per score row.")
        unique_labels, categorical_codes = np.unique(labels, return_inverse=True)
        cmap = plt.get_cmap("tab20", max(1, len(unique_labels)))
        scatter = ax.scatter(scores[:, 0], scores[:, 1], c=categorical_codes, s=10, alpha=0.75, cmap=cmap, vmin=-0.5, vmax=len(unique_labels)-0.5)
        cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(len(unique_labels)))
        cbar.ax.set_yticklabels([str(v) for v in unique_labels])
        cbar.set_label("Cluster label (noise is -1 for DBSCAN/HDBSCAN)")
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def plot_explained_variance(pca_model, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ratio = np.asarray(pca_model.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(np.arange(1, len(ratio) + 1), np.cumsum(ratio), marker="o")
    ax.set(xlabel="Number of principal components", ylabel="Cumulative explained variance", ylim=(0, 1.05))
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)
