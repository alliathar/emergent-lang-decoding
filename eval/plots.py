"""Plots for runs: training curves, symbol/attribute heatmaps and image grids.

Uses the non-interactive "Agg" backend so plotting works on machines with no
display (e.g. Kaggle) and never pops up windows during a sweep.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_training_curves(metrics: pd.DataFrame, chance: float, path: Path) -> None:
    """Accuracy and loss per epoch, with a dashed line at chance accuracy."""
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(10, 4))
    ax_acc.plot(metrics["epoch"], metrics["train_acc"], label="train")
    ax_acc.plot(metrics["epoch"], metrics["val_acc"], label="val")
    ax_acc.axhline(chance, color="grey", linestyle="--", label="chance")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", ylim=(0, 1.02), title="Accuracy")
    ax_acc.legend()
    ax_loss.plot(metrics["epoch"], metrics["train_loss"], label="train")
    ax_loss.plot(metrics["epoch"], metrics["val_loss"], label="val")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy", title="Loss")
    ax_loss.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_heatmap(table: pd.DataFrame, path: Path, title: str, vmax: float | None = 1.0) -> None:
    """Annotated heatmap of a DataFrame (rows x columns)."""
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * values.shape[1] + 2), max(2.5, 0.45 * values.shape[0] + 1.5)))
    im = ax.imshow(np.nan_to_num(values), cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(values.shape[1]), table.columns, rotation=60, ha="right")
    ax.set_yticks(range(values.shape[0]), [str(i) for i in table.index])
    ax.set_xlabel(table.columns.name or "")
    ax.set_ylabel(table.index.name or "")
    if values.size <= 400:  # numbers become unreadable on big tables
        for (i, j), v in np.ndenumerate(values):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if v > 0.6 * (vmax or values.max()) else "white")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_image_grid(images: np.ndarray, titles: list[str], path: Path, n_cols: int = 10) -> None:
    """images: (N, 3, 64, 64) uint8 or float. One small titled panel per image."""
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    title_lines = max((t.count("\n") + 1 for t in titles), default=1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, (1.5 + 0.15 * title_lines) * n_rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, title in zip(axes.flat, images, titles):
        ax.imshow(np.transpose(image, (1, 2, 0)))
        ax.set_title(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
