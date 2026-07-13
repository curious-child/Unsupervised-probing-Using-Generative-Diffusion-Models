from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg


def _crop_white_margin(image: np.ndarray, threshold: int = 250, pad: int = 8) -> np.ndarray:
    mask = np.any(image[:, :, :3] < threshold, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return image

    row0 = max(rows[0] - pad, 0)
    row1 = min(rows[-1] + pad + 1, image.shape[0])
    col0 = max(cols[0] - pad, 0)
    col1 = min(cols[-1] + pad + 1, image.shape[1])
    return image[row0:row1, col0:col1]


def render_source_figure(source_fig, dpi: int = 220, crop: bool = True) -> np.ndarray:
    source_fig.set_dpi(dpi)
    canvas = FigureCanvasAgg(source_fig)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba()).copy()
    plt.close(source_fig)
    return _crop_white_margin(image) if crop else image


def add_panel_image(ax, source_fig, label: str | None = None, dpi: int = 220) -> None:
    image = render_source_figure(source_fig, dpi=dpi)
    ax.imshow(image)
    ax.set_axis_off()
    if label:
        ax.text(
            -0.02,
            1.02,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="bottom",
            ha="left",
        )


def save_panel_grid(
    panel_figs: Sequence,
    output_base: Path,
    nrows: int,
    ncols: int,
    figsize: tuple[float, float],
    labels: Sequence[str] | None = None,
    dpi: int = 600,
    wspace: float = 0.05,
    hspace: float = 0.08,
    legend_handles: Sequence | None = None,
    legend_labels: Sequence[str] | None = None,
    legend_kwargs: dict | None = None,
) -> None:
    if len(panel_figs) != nrows * ncols:
        raise ValueError(f"expected {nrows * ncols} panels, got {len(panel_figs)}")

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for index, source_fig in enumerate(panel_figs):
        row, col = divmod(index, ncols)
        label = labels[index] if labels else None
        add_panel_image(axs[row, col], source_fig, label=label, dpi=dpi)

    top = 0.90 if legend_handles and legend_labels else 0.98
    fig.subplots_adjust(left=0.02, right=0.98, top=top, bottom=0.02, wspace=wspace, hspace=hspace)
    if legend_handles and legend_labels:
        kwargs = {
            "loc": "upper center",
            "bbox_to_anchor": (0.52, 1.01),
            "ncol": len(legend_labels),
            "fontsize": 9,
            "handlelength": 1.0,
            "columnspacing": 1.2,
        }
        if legend_kwargs:
            kwargs.update(legend_kwargs)
        fig.legend(legend_handles, legend_labels, **kwargs)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), dpi=dpi,bbox_inches="tight")
    plt.close(fig)
