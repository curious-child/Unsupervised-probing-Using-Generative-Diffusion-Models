"""Paper Fig. 2: topology generalization panels."""

from __future__ import annotations

import argparse
import gc

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    DATASETS,
    GRAPH_TYPES,
    add_common_args,
    dynamics_filename,
    ensure_output,
    graph_name,
    load_dynamic_record,
    parameters,
    select_one_or_all,
    spdata_source_path,
)
from figure_composer import save_panel_grid
from evaluation_and_analysis.diffusion_model_uncertainy import uncertainty_ews


RAW_COLOR = "#0F4D92"
PRED_COLOR = "#B64342"
TRANSITION_COLOR = "#B64342"
MPV_COLOR = "#B64342"
GRAPH_DATA_FALLBACKS = {
    "BA": "barabasi_albert_30_0",
    "ER": "erdos_renyi_50_0",
    "WS": "small-world_70_0",
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
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
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


def load_panel_data(ews_root, source_root, dataset_type: str, data_trend: str, graph_type: str) -> dict:
    data_name = dynamics_filename(dataset_type, data_trend)
    graph = graph_name(graph_type)
    source_path = spdata_source_path(source_root, dataset_type, graph, data_name)
    if not source_path.exists() and graph_type in GRAPH_DATA_FALLBACKS:
        source_path = spdata_source_path(source_root, dataset_type, GRAPH_DATA_FALLBACKS[graph_type], data_name)
    time_data, ys_dynamic = load_dynamic_record(source_path)
    sample_window_step, _ = parameters(dataset_type)

    model_root = ews_root / "ews_generalization" / "graph" / dataset_type
    cache_file = model_root / f"{graph_type}_{data_trend}.pt"
    result = uncertainty_ews(
        model_save_file=model_root,
        data_file=source_path,
        dynamic_type=dataset_type,
        cache_path=cache_file,
        sample_window_step=None if cache_file.exists() else sample_window_step,
        force_recompute=False,
    )
    ts, ys = sample_series(time_data, ys_dynamic, result["sampling_t"])
    data = {
        "ts": ts,
        "ys_mean": ys.mean(axis=1),
        "pred_mean": np.asarray(result["pred_mean"], dtype=float),
        "ews": np.asarray(result["ews"], dtype=float),
        "ews_ts": np.asarray(result["time_points"]),
        "dynamic_type": dataset_type,
        "data_trend": data_trend,
        "graph_type": graph_type,
    }
    del result
    gc.collect()
    return data


def plot_ews_compare(panel_data: dict, axs=None):
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(6, 6), gridspec_kw={"hspace": 0.00})
        owns_figure = True
    else:
        axs = np.asarray(axs, dtype=object).ravel()
        if len(axs) != 2:
            raise ValueError("plot_ews_compare expects 2 axes")
        fig = axs[0].figure
        owns_figure = False

    ts = panel_data["ts"]
    ys_mean = panel_data["ys_mean"]
    ews_ts = panel_data["ews_ts"]
    pred_mean = panel_data["pred_mean"]
    ews = panel_data["ews"]

    axs[0].plot(ts, ys_mean, color=RAW_COLOR, linewidth=2)
    #axs[0].plot(ews_ts[: len(pred_mean)], pred_mean, color=PRED_COLOR, linewidth=2)
    # uncertainty = np.sqrt(ews)
    # axs[0].fill_between(
    #     ews_ts[: len(pred_mean)],
    #     pred_mean - uncertainty,
    #     pred_mean + uncertainty,
    #     color=PRED_COLOR,
    #     alpha=0.2,
    #     linewidth=0,
    # )
    axs[0].set_ylabel("State")
    #axs[0].legend(["Original data", "Predicted future"], loc="best", frameon=False, fontsize=8)

    axs[1].plot(ews_ts[: len(ews)], ews, "o", color=PRED_COLOR, markersize=3.0, linewidth=0)
    axs[1].set_ylabel("MPV")
    axs[1].set_xlabel("Time")
    axs[1].sharex(axs[0])

    true_time = transition_time(
        ts,
        ys_mean,
        dataset_type=panel_data["dynamic_type"],
        data_trend=panel_data["data_trend"],
    )
  #  min_mpv_time = ews_ts[int(np.argmin(ews))]
    for ax in axs:
        ax.axvline(x=true_time, color=TRANSITION_COLOR, linestyle="--", linewidth=0.9, alpha=0.75)
       # ax.axvline(x=min_mpv_time, color=MPV_COLOR, linestyle="--", linewidth=1, alpha=0.7)
        ax.tick_params(length=3, width=0.8)
        ax.margins(x=0)
    axs[0].tick_params(labelbottom=False)

    if owns_figure:
        fig.tight_layout()
    return fig


def make_panel_figure(ews_root, source_root, dataset_type: str, data_trend: str, graph_type: str):
    return plot_ews_compare(load_panel_data(ews_root, source_root, dataset_type, data_trend, graph_type))


def build_fig2(ews_root, source_root, output_dir, data_trend: str, graph_types: list[str]) -> None:
    set_nature_style(font_size=14)
    panel_figs = [
        make_panel_figure(ews_root, source_root, dataset_type, data_trend, graph_type)
        for dataset_type in DATASETS
        for graph_type in graph_types
    ]
    labels = tuple(chr(ord("a") + index) for index in range(len(panel_figs)))
    save_panel_grid(
        panel_figs,
        output_dir / f"fig2_{data_trend}",
        nrows=len(DATASETS),
        ncols=len(graph_types),
        figsize=(9.8, 9.8),
        labels=labels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--graph-type", choices=GRAPH_TYPES)
    args = parser.parse_args()
    output_dir = ensure_output(args.output_dir)
    graph_types = select_one_or_all(args.graph_type, GRAPH_TYPES)
    build_fig2(args.ews_root, args.source_root, output_dir, args.trend, graph_types)


if __name__ == "__main__":
    main()
