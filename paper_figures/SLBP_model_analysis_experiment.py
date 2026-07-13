"""Paper Fig. 6: SLBP uncertainty interpretation panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common import TRENDS, add_common_args, ensure_output, load_dynamic_record, slbp_source_path
from evaluation_and_analysis.diffusion_model_uncertainy import (
    slbp_gx_analysis,
    slbp_raw_window_variance,
    slbp_sampling_analysis,
)


DIFFUSION_MODEL_NAME = "dataset_w200p200st100"
PREG_MODEL_NAME = "dataset_w200p200st100"
TOTAL_TIME = "1000000.0"
D_VALUE = "1e-05"
DIFFUSION_CACHE_SUBDIR = "T_1000000.0_D1e-05"
PRED_DIM = 0
SAMPLE_WINDOW_STEP = 10


def set_pub_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )


def buishand_like_transition_time(time_values, state_values):
    time_values = np.asarray(time_values, dtype=float)
    state_values = np.asarray(state_values, dtype=float)
    if len(time_values) <= 10 or len(state_values) <= 10:
        return np.nan
    slopes = np.abs(state_values[10:] - state_values[:-10]) / 10
    return float(time_values[int(np.argmax(slopes))])


def load_panel_data(ews_root: Path, source_root: Path, data_trend: str) -> dict:
    data_path = slbp_source_path(source_root, TOTAL_TIME, data_trend, D_VALUE, test_data=False)
    time_data, torch_time_series = load_dynamic_record(data_path)

    diffusion_root = ews_root / "NsDiff_trends"
    preg_root = ews_root / "NsDiff_preg"
    sampling = slbp_sampling_analysis(
        model_root=diffusion_root,
        model_name=DIFFUSION_MODEL_NAME,
        torch_time_series=torch_time_series,
        time_data=time_data,
        data_trend=data_trend,
        pred_dim=PRED_DIM,
        sample_window_step=SAMPLE_WINDOW_STEP,
        cache_subdir=DIFFUSION_CACHE_SUBDIR,
        allow_unavailable=True,
    )
    if not sampling["available"]:
        print("Fig.6 sampling unavailable for {}: {}".format(data_trend, sampling["reason"]))

    diffusion_gx = slbp_gx_analysis(
        model_root=diffusion_root,
        model_name=DIFFUSION_MODEL_NAME,
        torch_time_series=torch_time_series,
        time_data=time_data,
        data_trend=data_trend,
        pred_dim=PRED_DIM,
        sample_window_step=SAMPLE_WINDOW_STEP,
        cache_subdir=DIFFUSION_CACHE_SUBDIR,
        windows=sampling["windows"],
        pred_len=sampling["pred_len"],
        sampling_t=sampling["sampling_t"],
    )
    preg_gx = slbp_gx_analysis(
        model_root=preg_root,
        model_name=PREG_MODEL_NAME,
        torch_time_series=torch_time_series,
        time_data=time_data,
        data_trend=data_trend,
        pred_dim=PRED_DIM,
        sample_window_step=SAMPLE_WINDOW_STEP,
        windows=sampling["windows"],
        pred_len=sampling["pred_len"],
        sampling_t=sampling["sampling_t"],
    )
    raw_variance = slbp_raw_window_variance(
        torch_time_series=torch_time_series,
        time_data=time_data,
        windows=sampling["windows"],
        sampling_t=sampling["sampling_t"],
        sample_window_step=SAMPLE_WINDOW_STEP,
        pred_dim=PRED_DIM,
    )

    plot_time = np.asarray(time_data[::1000], dtype=float)
    plot_state = torch_time_series[::1000, PRED_DIM].cpu().detach().numpy()
    transition_time = buishand_like_transition_time(plot_time[1000:], plot_state[1000:])

    return {
        "trend": data_trend,
        "time_data": np.asarray(time_data, dtype=float),
        "state_time": plot_time,
        "state": plot_state,
        "transition_time": transition_time,
        "sampling": sampling,
        "diffusion_gx": diffusion_gx,
        "preg_gx": preg_gx,
        "raw_variance": raw_variance,
    }


def _plot_unavailable(ax, reason: str) -> None:
    ax.text(
        0.5,
        0.5,
        "unavailable",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#777777",
        fontsize=9,
    )
    if reason:
        ax.text(
            0.5,
            0.18,
            "see console",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#999999",
            fontsize=7,
        )


def plot_panel_column(axs, panel_data: dict, show_ylabel: bool) -> None:
    color_state = "#0F4D92"
    color_sampling = "#B64342"
    color_dim = "#9A4D8E"
    color_gx_diffusion = "#B64342"
    color_gx_preg = "#42949E"
    color_raw = "#4D4D4D"
    transition_time = panel_data["transition_time"]

    axs[0].plot(panel_data["state_time"], panel_data["state"], ".", color=color_state, markersize=2.1)
    axs[0].set_title(panel_data["trend"].capitalize(), fontsize=9.2, pad=3)

    sampling = panel_data["sampling"]
    if sampling["available"]:
        axs[1].plot(sampling["time_points"], sampling["mpv"], "-", color=color_sampling, linewidth=1.25)
        axs[2].plot(
            sampling["time_points"],
            sampling["intrinsic_dimension"],
            "-",
            color=color_dim,
            linewidth=1.25,
        )
    else:
        _plot_unavailable(axs[1], sampling["reason"])
        _plot_unavailable(axs[2], sampling["reason"])

    diffusion_gx = panel_data["diffusion_gx"]
    preg_gx = panel_data["preg_gx"]
    axs[3].plot(
        diffusion_gx["time_points"],
        diffusion_gx["gx_mpv"],
        "-",
        color=color_gx_diffusion,
        linewidth=1.2,
        label="Joint estimator",
    )
    axs[3].plot(
        preg_gx["time_points"],
        preg_gx["gx_mpv"],
        "-",
        color=color_gx_preg,
        linewidth=1.2,
        label="Variance-only",
    )
    axs[3].legend(loc="best", fontsize=6.8, handlelength=1.5)

    raw_variance = panel_data["raw_variance"]
    axs[4].plot(
        raw_variance["time_points"],
        raw_variance["variance"],
        "-",
        color=color_raw,
        linewidth=1.2,
    )

    labels = ("State", "MPV", "Dimension", "Variance Estimator", "Variance")
    for row, ax in enumerate(axs):
        if show_ylabel:
            ax.set_ylabel(labels[row])
        ax.axvline(transition_time, color="#B64342", linestyle="--", linewidth=0.85, alpha=0.75)
        ax.tick_params(axis="both", labelsize=7.6, width=0.7, length=3)
        ax.margins(x=0.01)
    axs[-1].set_xlabel("Time")


def build_fig6(ews_root: Path, source_root: Path, output_dir: Path, trends: tuple[str, ...] = TRENDS) -> None:
    set_pub_style()
    panel_data = [load_panel_data(ews_root, source_root, trend) for trend in trends]
    fig, axes = plt.subplots(
        5,
        len(panel_data),
        figsize=(8.2, 7.0),
        sharex="col",
        gridspec_kw={"hspace": 0.08, "wspace": 0.16},
    )
    if len(panel_data) == 1:
        axes = np.asarray(axes).reshape(5, 1)

    for col, data in enumerate(panel_data):
        plot_panel_column(axes[:, col], data, show_ylabel=(col == 0))
        axes[-1, col].set_xlim(-0.05, float(data["time_data"][-1]) + 0.05)
        for row in range(4):
            axes[row, col].tick_params(labelbottom=False)

    for col, label in enumerate(("a", "b", "c", "d")[: len(panel_data)]):
        axes[0, col].text(
            -0.18,
            1.08,
            label,
            transform=axes[0, col].transAxes,
            fontsize=11.2,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
   # fig.savefig(output_dir / "fig6_SLBP_model_analysis.svg", bbox_inches="tight")
    fig.savefig(output_dir / "fig6_SLBP_model_analysis.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig6_SLBP_model_analysis.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    output_dir = ensure_output(args.output_dir)
    build_fig6(args.ews_root, args.source_root, output_dir, trends=TRENDS)


if __name__ == "__main__":
    main()
