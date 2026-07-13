"""Paper Fig. 9: SLBP smoothed-input false-collapse check."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import add_common_args, ensure_output, load_dynamic_record, slbp_source_path
from evaluation_and_analysis.diffusion_model_uncertainy import (
    build_slbp_sensitivity_windows,
    load_sensitivity_model,
    run_slbp_gx_cache_for_fig6,
    summarize_slbp_gx_for_fig6,
    torch_data_preprocessing_like_slbp,
)


TOTAL_TIME = "1000000.0"
MODEL_NAME = "dataset_w200p200st100"
NOISE_LEVELS = ("0.0001", "1e-05", "1e-06")
TRENDS = ("increase", "decrease")
PRED_DIM = 0
SAMPLE_WINDOW_STEP = 10
X_MAX = 1000000.0


def set_pub_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.4,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )


def causal_moving_average_smooth_torch(timeseries_tensor, window=21, time_dim=0):
    if window < 1:
        raise ValueError("window must be >= 1.")
    if not torch.is_floating_point(timeseries_tensor):
        timeseries_tensor = timeseries_tensor.float()

    data = timeseries_tensor.movedim(time_dim, 0)
    smoothed = torch.empty_like(data)
    cumsum = torch.cumsum(data, dim=0)
    for idx in range(data.shape[0]):
        start = max(0, idx - window + 1)
        if start == 0:
            total = cumsum[idx]
            count = idx + 1
        else:
            total = cumsum[idx] - cumsum[start - 1]
            count = window
        smoothed[idx] = total / count
    return smoothed.movedim(0, time_dim)


def cache_path(ews_root: Path, noise: str, trend: str, smooth: bool) -> Path:
    subdir = "smooth_SLPB_T_1000000.0" if smooth else "SLPB_1000000.0"
    return ews_root / "NsDiff_trends" / "datas" / subdir / f"gx_D_{noise}_{trend}.pt"


def load_record(source_root: Path, noise: str, trend: str) -> dict:
    time_data, torch_time_series = load_dynamic_record(
        slbp_source_path(source_root, TOTAL_TIME, trend, noise, test_data=False)
    )
    return {"time_data": time_data, "torch_time_series": torch_time_series}


def build_smoothed_inputs(torch_time_series, time_data, windows: int, sampling_t: int, sample_window_step: int):
    sampled_series = torch_data_preprocessing_like_slbp(torch_time_series, sampling_t=sampling_t)
    sampled_time = torch_data_preprocessing_like_slbp(time_data, sampling_t=sampling_t, return_numpy=True)
    smooth_window = max(3, windows // 5)

    smooth_series = causal_moving_average_smooth_torch(
        sampled_series,
        window=smooth_window,
        time_dim=0,
    ).to(sampled_series.dtype)

    raw_windows = sampled_series.unfold(0, windows, sample_window_step).permute(0, 2, 1)
    smooth_windows = causal_moving_average_smooth_torch(
        raw_windows,
        window=smooth_window,
        time_dim=1,
    ).unbind(0)
    time_points = sampled_time[windows - 1 :: sample_window_step]
    return smooth_windows, sampled_time, smooth_series, time_points


def load_or_run_gx(model, input_datas, cache_file: Path, device, pred_dim: int):
    gx_list = run_slbp_gx_cache_for_fig6(
        model=model,
        input_datas=input_datas,
        cache_path=cache_file,
        device=device,
        pred_dim=pred_dim,
    )
    return np.asarray(summarize_slbp_gx_for_fig6(gx_list, pred_dim=pred_dim), dtype=float)


def load_panel_data(ews_root: Path, source_root: Path) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, method_config, _loaded_net_param = load_sensitivity_model(
        ews_root / "NsDiff_trends",
        MODEL_NAME,
        device=device,
    )
    dataset_config = method_config.get("dataset", {})
    windows = int(dataset_config["windows"])
    pred_len = int(dataset_config["pred_len"])
    sampling_t = int(dataset_config["sampling_t"])

    panels = []
    for trend in TRENDS:
        for noise in NOISE_LEVELS:
            record = load_record(source_root, noise, trend)
            input_datas, _pred_datas, time_points = build_slbp_sensitivity_windows(
                torch_time_series=record["torch_time_series"],
                time_data=record["time_data"],
                windows=windows,
                pred_len=pred_len,
                sampling_t=sampling_t,
                sample_window_step=SAMPLE_WINDOW_STEP,
            )
            smooth_inputs, smooth_state_time, smooth_state, smooth_time_points = build_smoothed_inputs(
                torch_time_series=record["torch_time_series"],
                time_data=record["time_data"],
                windows=windows,
                sampling_t=sampling_t,
                sample_window_step=SAMPLE_WINDOW_STEP,
            )

            raw_mpv = load_or_run_gx(
                model=model,
                input_datas=input_datas,
                cache_file=cache_path(ews_root, noise, trend, smooth=False),
                device=device,
                pred_dim=PRED_DIM,
            )
            smooth_mpv = load_or_run_gx(
                model=model,
                input_datas=smooth_inputs,
                cache_file=cache_path(ews_root, noise, trend, smooth=True),
                device=device,
                pred_dim=PRED_DIM,
            )

            panels.append(
                {
                    "trend": trend,
                    "noise": noise,
                    "state_time": np.asarray(record["time_data"][::1000], dtype=float),
                    "state": record["torch_time_series"][::1000, PRED_DIM].detach().cpu().numpy(),
                    "smooth_state_time": np.asarray(smooth_state_time, dtype=float),
                    "smooth_state": smooth_state[:, PRED_DIM].detach().cpu().numpy(),
                    "mpv_time": np.asarray(time_points[: len(raw_mpv)], dtype=float),
                    "raw_mpv": raw_mpv,
                    "smooth_mpv_time": np.asarray(smooth_time_points[: len(smooth_mpv)], dtype=float),
                    "smooth_mpv": smooth_mpv,
                }
            )
    return panels


def style_axis(ax, show_xlabel: bool = False) -> None:
    ax.tick_params(axis="both", labelsize=7.3, width=0.7, length=3)
    ax.margins(x=0.01)
    if show_xlabel:
        ax.set_xlabel("Time")
    else:
        ax.tick_params(labelbottom=False)


def plot_panel(fig, spec, panel: dict, label: str, show_ylabel: bool, show_xlabel: bool):
    sub = spec.subgridspec(2, 1, hspace=0.04)
    ax_state = fig.add_subplot(sub[0])
    ax_mpv = fig.add_subplot(sub[1], sharex=ax_state)

    raw_color = "#0F4D92"
    smooth_color = "#E28E2C"

    raw_handle, = ax_state.plot(panel["state_time"], panel["state"], "o", color=raw_color, markersize=2, label="Raw")
    smooth_handle, = ax_state.plot(
        panel["smooth_state_time"],
        panel["smooth_state"],
        ".-",
        color=smooth_color,
        linewidth=1.0,
        markersize=1.3,
        label="Smoothed",
    )
    ax_mpv.plot(panel["mpv_time"], panel["raw_mpv"], ".", color=raw_color, markersize=1.3)
    ax_mpv.plot(
        panel["smooth_mpv_time"],
        panel["smooth_mpv"],
        ".-",
        color=smooth_color,
        linewidth=1.0,
        markersize=1.3,
    )

    if show_ylabel:
        ax_state.set_ylabel("State")
        ax_mpv.set_ylabel("MPV")

    ax_state.text(-0.16, 1.08, label, transform=ax_state.transAxes, fontsize=10.6, fontweight="bold", va="bottom")
    style_axis(ax_state)
    style_axis(ax_mpv, show_xlabel=show_xlabel)
    ax_mpv.set_xlim(-0.05, X_MAX)
    return raw_handle, smooth_handle


def build_fig9(ews_root: Path, source_root: Path, output_dir: Path) -> None:
    set_pub_style()
    panels = load_panel_data(ews_root, source_root)
    fig = plt.figure(figsize=(9.0, 5.8))
    outer = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.22)
    labels = iter("abcdef")
    legend_handles = None

    for idx, panel in enumerate(panels):
        row = 0 if panel["trend"] == "increase" else 1
        col = NOISE_LEVELS.index(panel["noise"])
        handles = plot_panel(
            fig,
            outer[row, col],
            panel,
            next(labels),
            show_ylabel=(col == 0),
            show_xlabel=(row == 1),
        )
        if legend_handles is None:
            legend_handles = handles

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            ["Raw", "Smoothed"],
            loc="upper center",
            bbox_to_anchor=(0.52, 1.01),
            ncol=2,
            fontsize=7.6,
            handlelength=1.6,
        )

    output_base = output_dir / "fig9_SLBP_smoothing_false_collapse"
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    output_dir = ensure_output(args.output_dir)
    build_fig9(args.ews_root, args.source_root, output_dir)


if __name__ == "__main__":
    main()
