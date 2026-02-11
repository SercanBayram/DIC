from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def save_quiver_plot(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    score: np.ndarray,
    output_path: Path,
    title: str,
) -> Path:
    output_path = Path(output_path)

    mag = np.sqrt(np.nan_to_num(u, nan=0.0) ** 2 + np.nan_to_num(v, nan=0.0) ** 2)
    valid = ~np.isnan(u) & ~np.isnan(v)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.imshow(image, cmap="gray")
    q = ax.quiver(
        x[valid],
        y[valid],
        u[valid],
        v[valid],
        mag[valid],
        cmap="turbo",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.0025,
    )
    ax.set_title(title)
    ax.set_axis_off()
    cbar = plt.colorbar(q, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("|displacement| (px)")

    ax.text(
        0.01,
        0.98,
        f"mean score={np.nanmean(score):.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
