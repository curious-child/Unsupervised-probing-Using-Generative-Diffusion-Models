"""Paper Fig. 8: SLBP training-source comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common import add_common_args, ensure_output, load_dynamic_record, slbp_source_path
from evaluation_and_analysis.diffusion_model_uncertainy import slbp_direct_model_cache_analysis


TOTAL_TIME = "1000000.0"
D_VALUE = "1e-05"
X_MAX = 1000000.0
PRED_DIM = 0
SAMPLE_WINDOW_STEP = 10
TRAIN_START = 0.0
TRAIN_END = 500000.0
MODEL_PANELS = (
    ("1000000.0_radio_0.5_decrease", "Decrease-Trained", ("decrease",)),
    ("1000000.0_radio_0.5_increase", "Increase-Trained", ("increase",)),
    ("1000000.0_N__radio_0.5_all", "Both-Trained", ("decrease", "increase")),
)
TRENDS = ("decrease", "increase")


def set_pub_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )


def load_slbp_record(source_root: Path, trend: str) -> dict:
    time_data, torch_time_series = load_dynamic_record(
        slbp_source_path(source_root, TOTAL_TIME, trend, D_VALUE, test_data=False)
    )
    return {
        "time_data": time_data,
        "torch_time_series": torch_time_series,
        "plot_time": np.asarray(time_data[::1000], dtype=float),
        "plot_state": torch_time_series[::1000, PRED_DIM].detach().cpu().numpy(),
    }


def gx_cache_path(model_dir: Path, trend: str) -> Path:
    return model_dir / "SLPB_1000000.0" / f"gx_D_{D_VALUE}_{trend}.pt"


def load_mpv(model_dir: Path, record: dict, trend: str) -> dict:
    result = slbp_direct_model_cache_analysis(
        model_save_file=model_dir,
        torch_time_series=record["torch_time_series"],
        time_data=record["time_data"],
        cache_path=gx_cache_path(model_dir, trend),
        pred_dim=PRED_DIM,
        sample_window_step=SAMPLE_WINDOW_STEP,
        cache_kind="gx",
    )
    return {
        "time": np.asarray(result["time_points"], dtype=float),
        "mpv": np.asarray(result["mpv"], dtype=float),
        "cache_path": result["cache_path"],
    }


def load_panel_data(ews_root: Path, source_root: Path) -> tuple[dict, list[dict]]:
    records = {trend: load_slbp_record(source_root, trend) for trend in TRENDS}
    panels = []
    for folder, label, shaded_trends in MODEL_PANELS:
        model_dir = ews_root / "NsDiff_dataset" / folder
        panels.append(
            {
                "label": label,
                "shaded_trends": set(shaded_trends),
                "mpv": {trend: load_mpv(model_dir, records[trend], trend) for trend in TRENDS},
            }
        )
    return records, panels


def add_training_span(ax, label: bool = True) -> None:
    ax.axvspan(TRAIN_START, TRAIN_END, color="#D8D8D8", alpha=0.32, linewidth=0)
    if label:
        y0, y1 = ax.get_ylim()
        ax.text(
            TRAIN_START + 0.03 * (TRAIN_END - TRAIN_START),
            y0 + 0.86 * (y1 - y0),
            "Training Data",
            fontsize=6.7,
            color="#4D4D4D",
            va="top",
            ha="left",
        )


def style_axis(ax, show_xlabel: bool = False) -> None:
    ax.tick_params(axis="both", labelsize=7.4, width=0.7, length=3)
    ax.margins(x=0.01)
    if show_xlabel:
        ax.set_xlabel("Time")
    else:
        ax.tick_params(labelbottom=False)


def plot_panel_column(fig, spec, records: dict, panel: dict, label: str, show_ylabel: bool) -> None:
    sub = spec.subgridspec(4, 1, hspace=0.04)
    axs = [fig.add_subplot(sub[row]) for row in range(4)]
    for row in range(1, 4):
        axs[row].sharex(axs[0])

    state_color = "#0F4D92"
    mpv_color = "#B64342"
    row_defs = (
        ("decrease", "state", "State"),
        ("decrease", "mpv", "MPV"),
        ("increase", "state", "State"),
        ("increase", "mpv", "MPV"),
    )

    for ax, (trend, kind, ylabel) in zip(axs, row_defs):
        if kind == "state":
            record = records[trend]
            ax.plot(record["plot_time"], record["plot_state"], ".", color=state_color, markersize=1.35)
            if trend in panel["shaded_trends"]:
                add_training_span(ax)
        else:
            mpv = panel["mpv"][trend]
            ax.plot(mpv["time"], mpv["mpv"], ".", color=mpv_color, markersize=2,alpha=0.8)
        if show_ylabel:
            ax.set_ylabel(ylabel)
        style_axis(ax, show_xlabel=(ax is axs[-1]))

    axs[0].text(-0.16, 1.08, label, transform=axs[0].transAxes, fontsize=10.8, fontweight="bold", va="bottom")
    axs[0].text(0.5, 1.08, panel["label"], transform=axs[0].transAxes, fontsize=9.0, va="bottom", ha="center")
    axs[-1].set_xlim(-0.05, X_MAX)


def build_fig8(ews_root: Path, source_root: Path, output_dir: Path) -> None:
    set_pub_style()
    records, panels = load_panel_data(ews_root, source_root)
    fig = plt.figure(figsize=(8.8, 6.8))
    outer = fig.add_gridspec(1, 3, wspace=0.22)

    for col, panel in enumerate(panels):
        plot_panel_column(fig, outer[col], records, panel, chr(ord("a") + col), show_ylabel=(col == 0))

    output_base = output_dir / "fig8_SLBP_train_source_comparison"
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    output_dir = ensure_output(args.output_dir)
    build_fig8(args.ews_root, args.source_root, output_dir)


if __name__ == "__main__":
    main()
