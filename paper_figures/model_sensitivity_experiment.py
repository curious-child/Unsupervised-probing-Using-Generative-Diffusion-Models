"""Paper Fig. 4: SLBP model-parameter sensitivity panels."""

from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import add_common_args, ensure_output, load_dynamic_record, slbp_source_path
from figure_composer import save_panel_grid
from evaluation_and_analysis.diffusion_model_uncertainy import slbp_sensitivity_ews


PRED_LENS = (200, 500, 1000)
WINDOW_LENS = (200, 500, 1000)
COLORS = ("#0F4D92", "#42949E", "#9A4D8E")
LINESTYLES = ("-", "--", ":")


def set_nature_style(font_size: int = 10) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": font_size,
            "axes.labelsize": font_size + 1,
            "axes.titlesize": font_size + 2,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )


def transition_time(time: np.ndarray, data: np.ndarray) -> float:
    window_size = 10
    if len(data) <= window_size:
        return float(time[np.argmax(np.abs(data - np.mean(data)))])
    change_score = np.abs(data[window_size:] - data[:-window_size]) / window_size
    return float(time[int(np.argmax(np.abs(change_score)))])


def load_sensitivity_data(ews_root, source_root, data_trend: str, metric: str) -> dict:
    data_path = slbp_source_path(source_root, "1000000.0", data_trend, "1e-05", test_data=False)
    time_data, torch_time_series = load_dynamic_record(data_path)

    plt_ews_dict = {
        "ts": torch.as_tensor(time_data)[::1000].cpu().numpy(),
        "ys": torch.as_tensor(torch_time_series, dtype=torch.float32)[::1000, 0].cpu().numpy(),
        "pred_ews_ts": {},
        "pred_ews": {},
        "win_ews": {},
        "win_ews_ts": {},
    }

    for pred_len in PRED_LENS:
        model_name = f"dataset__w200p{pred_len}st100"
        result = slbp_sensitivity_ews(
            model_root=ews_root / "NsDiff_pred",
            model_name=model_name,
            torch_time_series=torch_time_series,
            time_data=time_data,
            data_trend=data_trend,
            pred_dim=0,
        )
        values = result["mpv"] if metric == "mpv" else result["prediction_error"]
        plt_ews_dict["pred_ews"][pred_len] = np.asarray(values, dtype=float)
        plt_ews_dict["pred_ews_ts"][pred_len] = np.asarray(result["time_points"])

    for window_len in WINDOW_LENS:
        model_name = f"dataset__w{window_len}p200st100"
        result = slbp_sensitivity_ews(
            model_root=ews_root / "NsDiff_windows",
            model_name=model_name,
            torch_time_series=torch_time_series,
            time_data=time_data,
            data_trend=data_trend,
            pred_dim=0,
        )
        values = result["mpv"] if metric == "mpv" else result["prediction_error"]
        plt_ews_dict["win_ews"][window_len] = np.asarray(values, dtype=float)
        plt_ews_dict["win_ews_ts"][window_len] = np.asarray(result["time_points"])

    return plt_ews_dict


def plot_sensitivity_panel(plt_ews_dict, metric_ylabel: str, axs=None):
    if axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(6, 10), gridspec_kw={"hspace": 0.00})
        owns_figure = True
    else:
        axs = np.asarray(axs, dtype=object).ravel()
        if len(axs) != 3:
            raise ValueError("plot_sensitivity_panel expects 3 axes")
        fig = axs[0].figure
        owns_figure = False

    ts = plt_ews_dict["ts"]
    ys = plt_ews_dict["ys"]
    axs[0].plot(ts, ys, ".", color="#0F4D92", linewidth=2)
    axs[0].set_ylabel("State")

    tipping_point_time = transition_time(ts[1000:], ys[1000:]) if len(ts) > 1010 else transition_time(ts, ys)

    for index, model_param in enumerate(plt_ews_dict["pred_ews_ts"].keys()):
        sample_timepoints = plt_ews_dict["pred_ews_ts"][model_param]
        values = plt_ews_dict["pred_ews"][model_param]
        axs[1].plot(
            sample_timepoints[: len(values)],
            values,
            color=COLORS[index],
            linestyle=LINESTYLES[index],
            alpha=0.8,
            label=f"Pred-len:{model_param}",
            linewidth=2,
        )
    axs[1].sharex(axs[0])
    axs[1].legend(loc="best", frameon=False, fontsize=10)
    axs[1].set_ylabel(metric_ylabel)

    for index, model_param in enumerate(plt_ews_dict["win_ews_ts"].keys()):
        sample_timepoints = plt_ews_dict["win_ews_ts"][model_param]
        values = plt_ews_dict["win_ews"][model_param]
        axs[2].plot(
            sample_timepoints[: len(values)],
            values,
            color=COLORS[index],
            linestyle=LINESTYLES[index],
            alpha=0.4,
            label=f"Window-len:{model_param}",
            linewidth=1,
        )
    axs[2].sharex(axs[0])
    axs[2].set_ylabel(metric_ylabel)
    axs[2].legend(loc="best", frameon=False, fontsize=7.5)
    axs[2].set_xlabel("Time")

    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)
    for ax in axs:
        ax.axvline(x=tipping_point_time, color="#B64342", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.set_xlim([-0.05, ts[-1] + 0.05])
        ax.tick_params(length=3, width=0.8)

    if owns_figure:
        fig.tight_layout()
    return fig


def make_panel_figure(ews_root, source_root, data_trend: str, metric: str):
    panel_data = load_sensitivity_data(ews_root, source_root, data_trend, metric=metric)
    metric_ylabel = "MPV" if metric == "mpv" else "Prediction Error"
    return plot_sensitivity_panel(panel_data, metric_ylabel=metric_ylabel)


def build_fig4(ews_root, source_root, output_dir, data_trend: str) -> None:
    set_nature_style(font_size=14)
    panel_figs = [
        make_panel_figure(ews_root, source_root, data_trend, metric="mpv"),
        make_panel_figure(ews_root, source_root, data_trend, metric="prediction_error"),
    ]
    save_panel_grid(
        panel_figs,
        output_dir / f"fig4_SLBP_sensitivity_{data_trend}",
        nrows=1,
        ncols=2,
        figsize=(10.0, 7.2),
        labels=("a", "b"),
        wspace=0.02,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    output_dir = ensure_output(args.output_dir)
    build_fig4(args.ews_root, args.source_root, output_dir, args.trend)


if __name__ == "__main__":
    main()
