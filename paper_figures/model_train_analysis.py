"""Paper Fig. 7: SLBP dataset-constant and mechanism-ablation panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common import TRENDS, add_common_args, ensure_output, load_dynamic_record, slbp_source_path
from evaluation_and_analysis.diffusion_model_uncertainy import slbp_direct_model_cache_analysis



TOTAL_TIME = "1000000.0"
TEST_D_VALUE = "1e-05"
PRED_DIM = 0
SAMPLE_WINDOW_STEP = 10
TRAIN_D_PANELS = (
    ("D_0.001", "Train D = 0.001"),
    ("D_0.0001", "Train D = 0.0001"),
    ("D_1e-05", "Train D = 1e-05"),
)
TRAIN_N_PANELS = (
    ("D_0.001_N_0.5", "N = 0.5"),
    ("D_0.001_N_2.5", "N = 2.5"),
    ("D_0.001_N_5", "N = 5"),
)
ABLATION_PANELS = (
    ("wo_gx", "w/o gx"),
    ("wo_fx", "w/o fx"),
    ("wo_UANS", "w/o UANS"),
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


def transition_time(time_values, state_values) -> float:
    time_values = np.asarray(time_values, dtype=float)
    state_values = np.asarray(state_values, dtype=float)
    window_size = max(10, len(state_values) // 500)
    if len(time_values) <= window_size:
        return float(time_values[np.argmax(np.abs(state_values - np.mean(state_values)))])
    change_score = np.abs(state_values[window_size:] - state_values[:-window_size]) / window_size
    return float(time_values[int(np.argmax(change_score)) + window_size])


def load_test_record(source_root: Path, trend: str):
    data_path = slbp_source_path(source_root, TOTAL_TIME, trend, TEST_D_VALUE, test_data=False)
    return load_dynamic_record(data_path)


def gx_cache(model_dir: Path, trend: str) -> Path:
    return model_dir / "SLPB_1000000.0" / f"gx_D_{TEST_D_VALUE}_{trend}.pt"


def sampling_cache(model_dir: Path, trend: str) -> Path:
    return model_dir / "SLPB_1000000.0" / f"D_{TEST_D_VALUE}_{trend}.pt"


def load_gx_panel(ews_root: Path, torch_time_series, time_data, trend: str, folder: str, title: str) -> dict:
    model_dir = ews_root / "NsDiff_dataset_contant" / folder
    result = slbp_direct_model_cache_analysis(
        model_save_file=model_dir,
        torch_time_series=torch_time_series,
        time_data=time_data,
        cache_path=gx_cache(model_dir, trend),
        pred_dim=PRED_DIM,
        sample_window_step=SAMPLE_WINDOW_STEP,
        cache_kind="gx",
    )
    return {"title": title, "mpv_time": np.asarray(result["time_points"]), "mpv": np.asarray(result["mpv"])}


def load_ablation_panel(ews_root: Path, torch_time_series, time_data, trend: str, folder: str, title: str) -> dict:
    model_dir = ews_root / "NsDiff_machine" / folder
    result = slbp_direct_model_cache_analysis(
        model_save_file=model_dir,
        torch_time_series=torch_time_series,
        time_data=time_data,
        cache_path=sampling_cache(model_dir, trend),
        pred_dim=PRED_DIM,
        sample_window_step=SAMPLE_WINDOW_STEP,
        cache_kind="sampling",
        compute_prediction_error=True,
    )
    return {
        "title": title,
        "mpv_time": np.asarray(result["time_points"]),
        "mpv": np.asarray(result["mpv"]),
        "error": np.asarray(result["prediction_error"]),
    }


def load_figure_data(ews_root: Path, source_root: Path, trend: str) -> dict:
    time_data, torch_time_series = load_test_record(source_root, trend)
    state_time = np.asarray(time_data[::1000], dtype=float)
    state = torch_time_series[::1000, PRED_DIM].cpu().detach().numpy()
    t_transition = transition_time(
        np.asarray(time_data, dtype=float),
        torch_time_series[:, PRED_DIM].cpu().detach().numpy(),
    )
    return {
        "trend": trend,
        "time_data": np.asarray(time_data, dtype=float),
        "state_time": state_time,
        "state": state,
        "transition_time": t_transition,
        "train_d": [load_gx_panel(ews_root, torch_time_series, time_data, trend, folder, title) for folder, title in TRAIN_D_PANELS],
        "train_n": [load_gx_panel(ews_root, torch_time_series, time_data, trend, folder, title) for folder, title in TRAIN_N_PANELS],
        "ablation": [load_ablation_panel(ews_root, torch_time_series, time_data, trend, folder, title) for folder, title in ABLATION_PANELS],
    }


def format_axis(ax, transition: float, show_xlabel: bool = False) -> None:
    ax.axvline(transition, color="#B64342", linestyle="--", linewidth=0.85, alpha=0.75)
    ax.tick_params(axis="both", labelsize=7.4, width=0.7, length=3)
    ax.margins(x=0.01)
    if show_xlabel:
        ax.set_xlabel("Time")
    else:
        ax.tick_params(labelbottom=False)


def plot_state_mpv_cell(fig, spec, panel: dict, figure_data: dict, label: str, show_ylabel: bool) -> None:
    sub = spec.subgridspec(2, 1, hspace=0.03)
    ax_state = fig.add_subplot(sub[0])
    ax_mpv = fig.add_subplot(sub[1], sharex=ax_state)
    ax_state.plot(figure_data["state_time"], figure_data["state"], ".", color="#0F4D92", markersize=1.5)
    ax_mpv.plot(panel["mpv_time"], panel["mpv"], ".", color="#B64342", markersize=2.0,alpha=0.8)
    ax_state.text(-0.15, 1.04, label, transform=ax_state.transAxes, fontsize=10.2, fontweight="bold", va="bottom")
    if show_ylabel:
        ax_state.set_ylabel("State")
        ax_mpv.set_ylabel("MPV")
    format_axis(ax_state, figure_data["transition_time"])
    format_axis(ax_mpv, figure_data["transition_time"])


def plot_mpv_cell(fig, spec, panel: dict, figure_data: dict, label: str, show_ylabel: bool) -> None:
    ax = fig.add_subplot(spec)
    ax.plot(panel["mpv_time"], panel["mpv"], ".", color="#B64342", markersize=2.0,alpha=0.8)
    ax.text(-0.15, 1.04, label, transform=ax.transAxes, fontsize=10.2, fontweight="bold", va="bottom")
    if show_ylabel:
        ax.set_ylabel("MPV")
    format_axis(ax, figure_data["transition_time"])


def plot_ablation_cell(fig, spec, panel: dict, figure_data: dict, label: str, show_ylabel: bool) -> None:
    sub = spec.subgridspec(2, 1, hspace=0.03)
    ax_mpv = fig.add_subplot(sub[0])
    ax_error = fig.add_subplot(sub[1], sharex=ax_mpv)
    ax_mpv.plot(panel["mpv_time"], panel["mpv"],  ".", color="#B64342", markersize=2.0,alpha=0.8)
    ax_error.plot(panel["mpv_time"][: len(panel["error"])], panel["error"], "-", color="#4D4D4D", linewidth=1.1)
    ax_mpv.text(-0.15, 1.04, label, transform=ax_mpv.transAxes, fontsize=10.2, fontweight="bold", va="bottom")
    if show_ylabel:
        ax_mpv.set_ylabel("MPV")
        ax_error.set_ylabel("Prediction Error")
    format_axis(ax_mpv, figure_data["transition_time"])
    format_axis(ax_error, figure_data["transition_time"], show_xlabel=True)


def build_fig7(ews_root: Path, source_root: Path, output_dir: Path, trend: str) -> None:
    set_pub_style()
    figure_data = load_figure_data(ews_root, source_root, trend)
    fig = plt.figure(figsize=(9.0, 7.0))
    outer = fig.add_gridspec(3, 3, hspace=0.23, wspace=0.22)
    labels = iter("abcdefghi")

    for col, panel in enumerate(figure_data["train_d"]):
        plot_state_mpv_cell(fig, outer[0, col], panel, figure_data, next(labels), show_ylabel=(col == 0))
    for col, panel in enumerate(figure_data["train_n"]):
        plot_mpv_cell(fig, outer[1, col], panel, figure_data, next(labels), show_ylabel=(col == 0))
    for col, panel in enumerate(figure_data["ablation"]):
        plot_ablation_cell(fig, outer[2, col], panel, figure_data, next(labels), show_ylabel=(col == 0))

    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 600
        fig.savefig(output_dir / f"fig7_SLBP_dataset_constant_{trend}.{suffix}", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    output_dir = ensure_output(args.output_dir)
    for trend in TRENDS:
        build_fig7(args.ews_root, args.source_root, output_dir, trend)


if __name__ == "__main__":
    main()
