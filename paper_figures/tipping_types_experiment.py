"""Paper Fig. 5: SLBP noise and observation-time parameter grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common import TRENDS, add_common_args, ensure_output, load_dynamic_record, slbp_source_path
from evaluation_and_analysis.diffusion_model_uncertainy import slbp_mpv_analysis


MODEL_NAME = "dataset_w200p200st100"
PRED_DIM = 0
NOISE_LEVELS = ("0.0005", "0.0001", "1e-05")
TIME_ROWS = (
    ("1000000.0", "SLPB_1000000.0", 10),
    ("50000.0", "SLPB_50000.0", 1),
)


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


def cache_file(ews_root: Path, cache_folder: str, d_value: str, trend: str) -> Path:
    return ews_root / "NsDiff_trends" / "datas" / cache_folder / f"D_{d_value}_{trend}.pt"


def load_panel_data(ews_root: Path, source_root: Path, trend: str, total_time: str, cache_folder: str, d_value: str, sample_window_step: int) -> dict:
    data_path = slbp_source_path(source_root, total_time, trend, d_value, test_data=False)
    time_data, torch_time_series = load_dynamic_record(data_path)
    mpv_result = slbp_mpv_analysis(
        model_root=ews_root / "NsDiff_trends",
        model_name=MODEL_NAME,
        torch_time_series=torch_time_series,
        time_data=time_data,
        cache_path=cache_file(ews_root, cache_folder, d_value, trend),
        pred_dim=PRED_DIM,
        sample_window_step=sample_window_step,
    )

    state_time = np.asarray(time_data[::1000], dtype=float)
    state = torch_time_series[::1000, PRED_DIM].cpu().detach().numpy()
    return {
        "total_time": total_time,
        "d_value": d_value,
        "trend": trend,
        "state_time": state_time,
        "state": state,
        "mpv_time": np.asarray(mpv_result["time_points"], dtype=float),
        "mpv": np.asarray(mpv_result["mpv"], dtype=float),
        "cache_path": mpv_result["cache_path"],
        "uncertainty_source": mpv_result["uncertainty_source"],
    }


def plot_panel(ax_state, ax_mpv, panel: dict, label: str, show_ylabel: bool, show_xlabel: bool) -> None:
    state_color = "#0F4D92"
    mpv_color = "#B64342"

    ax_state.plot(panel["state_time"], panel["state"], ".", color=state_color, markersize=1.7)
    ax_mpv.plot(panel["mpv_time"], panel["mpv"], "-", color=mpv_color, linewidth=1.15)

    for ax in (ax_state, ax_mpv):
        ax.tick_params(axis="both", labelsize=7.4, width=0.7, length=3)
        ax.margins(x=0.01)

    ax_state.text(
        -0.08,
        1.04,
        label,
        transform=ax_state.transAxes,
        fontsize=10.2,
        fontweight="bold",
        va="top",
        ha="left",
    )
   # ax_state.set_title(f"D = {panel['d_value']}", fontsize=9, pad=2)
    ax_state.tick_params(labelbottom=False)
    if show_ylabel:
        ax_state.set_ylabel("State")
        ax_mpv.set_ylabel("MPV")
    if show_xlabel:
        ax_mpv.set_xlabel("Time")
    else:
        ax_mpv.tick_params(labelbottom=False)


def build_fig5_for_trend(ews_root: Path, source_root: Path, output_dir: Path, trend: str) -> None:
    set_pub_style()
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(8.8, 6.1),
        gridspec_kw={"hspace": 0.16, "wspace": 0.20},
    )

    panel_index = 0
    for row_index, (total_time, cache_folder, sample_window_step) in enumerate(TIME_ROWS):
        state_row = row_index * 2
        mpv_row = state_row + 1
        for col_index, d_value in enumerate(NOISE_LEVELS):
            panel = load_panel_data(
                ews_root=ews_root,
                source_root=source_root,
                trend=trend,
                total_time=total_time,
                cache_folder=cache_folder,
                d_value=d_value,
                sample_window_step=sample_window_step,
            )
            label = chr(ord("a") + panel_index)
            plot_panel(
                axes[state_row, col_index],
                axes[mpv_row, col_index],
                panel,
                label=label,
                show_ylabel=(col_index == 0),
                show_xlabel=(row_index == len(TIME_ROWS) - 1),
            )
            axes[state_row, col_index].set_xlim(0, float(total_time))
            axes[mpv_row, col_index].set_xlim(0, float(total_time))
            panel_index += 1

   # fig.suptitle(trend, fontsize=11, y=0.995)
    base = output_dir / f"fig5_SLBP_parameter_grid_{trend}"
   # fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_fig5(ews_root: Path, source_root: Path, output_dir: Path, trends: tuple[str, ...] = TRENDS) -> None:
    for trend in trends:
        build_fig5_for_trend(ews_root, source_root, output_dir, trend)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    output_dir = ensure_output(args.output_dir)
    build_fig5(args.ews_root, args.source_root, output_dir, trends=TRENDS)


if __name__ == "__main__":
    main()
