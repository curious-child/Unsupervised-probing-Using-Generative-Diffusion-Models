"""Paper Fig. 3: dynamics generalization panels."""

from __future__ import annotations

import argparse
import gc

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from common import (
    DATASETS,
    add_common_args,
    dynamics_filename,
    dynamics_title,
    ensure_output,
    load_dynamic_record,
    parameters,
    spdata_source_path,
)
from figure_composer import save_panel_grid
from evaluation_and_analysis.diffusion_model_uncertainy import uncertainty_ews


RAW_COLOR = "#0F4D92"
MPV_COLOR = "#B64342"
TRANSITION_COLOR = "#B64342"
TRAINED_LABELS = {
    "biomass": "Biomass-trained",
    "neuronal": "Neuronal-trained",
    "SIS": "SIS-trained",
}
TRAINED_COLORS = {
    "biomass": "#B64342",
    "neuronal": "#42949E",
    "SIS": "#9A4D8E",
}


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
            "xtick.labelsize": font_size-2,
            "ytick.labelsize": font_size-2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )


def sample_series(time_data, ys_dynamic, sampling_t: float) -> tuple[np.ndarray, np.ndarray]:
    interval = max(1, int(sampling_t / 0.1))
    ts = torch.as_tensor(time_data)[::interval].cpu().numpy()
    ys = torch.as_tensor(ys_dynamic, dtype=torch.float32)[::interval, :].cpu().numpy()
    return ts, ys


def transition_time(time: np.ndarray, data: np.ndarray, dataset_type: str, data_trend: str) -> float:
    window_size = 10
    if len(data) <= window_size:
        return float(time[np.argmax(np.abs(data - np.mean(data)))])
    if dataset_type in ("biomass", "neuronal"):
        score = np.abs(data[window_size:] - data[:-window_size]) / window_size
        change_point_index = int(np.argmax(score)) + window_size // 2
    elif dataset_type == "SIS":
        rolling_mean = np.array([np.mean(data[i : i + window_size]) for i in range(len(data) - window_size)])
        if data_trend == "increase":
            candidates = np.argwhere(rolling_mean > 1e-2).flatten()
        else:
            candidates = np.argwhere(rolling_mean < 1e-2).flatten()
        change_point_index = int(candidates[0]) + window_size // 2 if len(candidates) else int(np.argmax(np.abs(np.diff(data))))
    else:
        raise ValueError(f"unknown dataset_type: {dataset_type}")
    return float(time[min(change_point_index, len(time) - 1)])


def load_panel_data(ews_root, source_root, dataset_type: str, data_trend: str) -> dict:
    data_name = dynamics_filename(dataset_type, data_trend)
    source_path = spdata_source_path(source_root, dataset_type, "barabasi_albert_30_0", data_name)
    time_data, ys_dynamic = load_dynamic_record(source_path)
    sample_window_step, sample_ts = parameters(dataset_type)

    model_signals = {}
    for trained_on in DATASETS:
        model_root = ews_root / "ews_generalization" / "dynamic" / trained_on
        cache_file = model_root / data_name
        result = uncertainty_ews(
            model_save_file=model_root,
            data_file=source_path,
            dynamic_type=dataset_type,
            cache_path=cache_file,
            sample_window_step=None if cache_file.exists() else sample_window_step,
            sampling_t=sample_ts,
            force_recompute=False,
        )
        model_signals[trained_on] = {
            "ews": np.asarray(result["ews"], dtype=float),
            "ews_ts": np.asarray(result["time_points"]),
        }
        del result
        gc.collect()

    ts, ys = sample_series(time_data, ys_dynamic, sample_ts)
    return {
        "ts": ts,
        "ys_mean": ys.mean(axis=1),
        "models": model_signals,
        "dynamic_type": dataset_type,
        "data_trend": data_trend,
    }


def plot_ews_compare(panel_data: dict, axs=None):
    if axs is None:
        fig, axs = plt.subplots(4, 1, figsize=(6, 10), gridspec_kw={"hspace": 0.00})
        owns_figure = True
    else:
        axs = np.asarray(axs, dtype=object).ravel()
        if len(axs) != 4:
            raise ValueError("plot_ews_compare expects 4 axes")
        fig = axs[0].figure
        owns_figure = False

    axs[0].set_title(dynamics_title(panel_data["dynamic_type"]))
    axs[0].plot(panel_data["ts"], panel_data["ys_mean"], color=RAW_COLOR, linewidth=2)
    axs[0].set_ylabel("State")

    for row, trained_on in enumerate(DATASETS, start=1):
        signal = panel_data["models"][trained_on]
        ews = signal["ews"]
        ews_ts = signal["ews_ts"]
        axs[row].plot(
            ews_ts[: len(ews)],
            ews,
            "o",
            color=TRAINED_COLORS.get(trained_on, MPV_COLOR),
            markersize=2.0,
            linewidth=0,
            label=TRAINED_LABELS[trained_on],
        )
        axs[row].set_ylabel("MPV")
        axs[row].sharex(axs[0])

    axs[-1].set_xlabel("Time")
    true_time = transition_time(
        panel_data["ts"],
        panel_data["ys_mean"],
        dataset_type=panel_data["dynamic_type"],
        data_trend=panel_data["data_trend"],
    )
    for ax in axs:
        ax.axvline(x=true_time, color=TRANSITION_COLOR, linestyle="--", linewidth=0.9, alpha=0.75)
        ax.tick_params(length=3, width=0.8)
        ax.margins(x=0)
    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)

    if owns_figure:
        fig.tight_layout()
    return fig


def make_panel_figure(ews_root, source_root, dataset_type: str, data_trend: str):
    return plot_ews_compare(load_panel_data(ews_root, source_root, dataset_type, data_trend))


def build_fig3(ews_root, source_root, output_dir, data_trend: str) -> None:
    set_nature_style(font_size=14)
    panel_figs = [
        make_panel_figure(ews_root, source_root, dataset_type, data_trend)
        for dataset_type in DATASETS
    ]
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=TRAINED_COLORS[trained_on], markersize=4)
        for trained_on in DATASETS
    ]
    legend_labels = [TRAINED_LABELS[trained_on] for trained_on in DATASETS]
    save_panel_grid(
        panel_figs,
        output_dir / f"fig3_{data_trend}",
        nrows=1,
        ncols=len(DATASETS),
        figsize=(11.2, 6.5),
        labels=("a", "b", "c"),
        legend_handles=legend_handles,
        legend_labels=legend_labels,
        legend_kwargs={"fontsize": 10.5},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    output_dir = ensure_output(args.output_dir)
    build_fig3(args.ews_root, args.source_root, output_dir, args.trend)


if __name__ == "__main__":
    main()
