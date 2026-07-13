"""Paper Fig. 1: diffusion-model uncertainty signals across network dynamics."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (
    DATASETS,
    EWS_ROOT,
    OUTPUT_ROOT,
    Source_ROOT,
    add_common_args,
    dynamics_filename,
    dynamics_title,
    ensure_output,
    load_dynamic_record,
    spdata_source_path,
)
from evaluation_and_analysis.diffusion_model_uncertainy import uncertainty_ews


FIG1_MODELS = ("NsDiff", "DiffSTG", "DiffusionTS", "TMDM")
MODEL_COLORS = {
    "NsDiff": "#B64342",
    "DiffSTG": "#42949E",
    "DiffusionTS": "#9A4D8E",
    "TMDM": "#E28E2C",
}
RAW_COLOR = "#0F4D92"
TRANSITION_COLOR = "#B64342"
BASELINE_FRACTION = 0.1
MIN_BASELINE_POINTS = 5


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


def graph_file(source_root: Path, graph: str) -> Path:
    return source_root / "test_graph" / f"{graph}.graphml"


def transition_time(time: np.ndarray, data: np.ndarray, dataset_type: str, data_trend: str) -> float:
    window_size = 10
    if len(data) <= window_size:
        return float(time[np.argmax(np.abs(data - np.mean(data)))])

    if dataset_type in ("biomass", "neuronal"):
        change_score = np.abs(data[window_size:] - data[:-window_size]) / window_size
        change_point_index = int(np.argmax(change_score))
    elif dataset_type == "SIS":
        rolling_mean = np.array([np.mean(data[i : i + window_size]) for i in range(len(data) - window_size)])
        if data_trend == "increase":
            candidates = np.argwhere(rolling_mean > 1e-2).flatten()
        else:
            candidates = np.argwhere(rolling_mean < 1e-2).flatten()
        change_point_index = int(candidates[0]) if len(candidates) else int(np.argmax(np.abs(np.diff(data))))
    else:
        raise ValueError(f"unknown dataset_type: {dataset_type}")
    return float(time[change_point_index])


def load_raw_trace(source_root: Path, dataset_type: str, data_trend: str, graph: str) -> dict:
    data_name = dynamics_filename(dataset_type, data_trend)
    source_path = spdata_source_path(source_root, dataset_type, graph, data_name)
    time_data, ys_dynamic = load_dynamic_record(source_path)
    ys = torch.as_tensor(ys_dynamic, dtype=torch.float32)
    if ys.ndim != 2:
        raise ValueError(f"{source_path} must contain ys_dynamic with shape [T, Node].")
    mean = ys.mean(dim=1).cpu().numpy()
    variance = ys.var(dim=1, unbiased=False).cpu().numpy()
    return {
        "data_file": source_path,
        "time": torch.as_tensor(time_data).cpu().numpy(),
        "mean": mean,
        "variance": variance,
        "transition_time": transition_time(
            torch.as_tensor(time_data).cpu().numpy(),
            mean,
            dataset_type=dataset_type,
            data_trend=data_trend,
        ),
    }


def load_model_signal(
    ews_root: Path,
    source_root: Path,
    model_name: str,
    dataset_type: str,
    data_file: Path,
    graph: str,
    force_recompute: bool = False,
) -> dict:
    model_dir = ews_root / "model_compare" / model_name / dataset_type
    if not model_dir.exists():
        raise FileNotFoundError(f"model folder not found: {model_dir}")

    result = uncertainty_ews(
        model_save_file=model_dir,
        data_file=data_file,
        dynamic_type=dataset_type,
        task_model=model_name if model_name == "DiffSTG" else None,
        graph_file=graph_file(source_root, graph) if model_name == "DiffSTG" else None,
        cache_path=model_dir,
        force_recompute=force_recompute,
    )
    signal = {
        "time": np.asarray(result["time_points"]),
        "ews": np.asarray(result["ews"], dtype=float),
        "cache_path": result["cache_path"],
        "sample_window_step": result["sample_window_step"],
    }
    del result
    gc.collect()
    return signal


def thin_for_plot(x: np.ndarray, *ys: np.ndarray, max_points: int = 1800) -> tuple[np.ndarray, ...]:
    if len(x) <= max_points:
        return (x, *ys)
    step = int(np.ceil(len(x) / max_points))
    return (x[::step], *(y[::step] for y in ys))


def relative_uncertainty(ews: np.ndarray) -> np.ndarray:
    ews = np.asarray(ews, dtype=float)
    if len(ews) == 0:
        return ews
    baseline_len = min(len(ews), max(MIN_BASELINE_POINTS, int(np.ceil(len(ews) * BASELINE_FRACTION))))
    baseline = np.nanmean(ews[:baseline_len])
    if not np.isfinite(baseline) or abs(baseline) < np.finfo(float).eps:
        return ews
    return ews / baseline


def collect_fig1_data(
    ews_root: Path,
    source_root: Path,
    data_trend: str,
    graph: str,
    models: tuple[str, ...],
    force_recompute: bool = False,
) -> dict:
    fig_data = {}
    for dataset_type in DATASETS:
        raw = load_raw_trace(source_root, dataset_type, data_trend, graph)
        model_signals = {}
        for model_name in models:
            model_signals[model_name] = load_model_signal(
                ews_root=ews_root,
                source_root=source_root,
                model_name=model_name,
                dataset_type=dataset_type,
                data_file=raw["data_file"],
                graph=graph,
                force_recompute=force_recompute,
            )
        fig_data[dataset_type] = {"raw": raw, "models": model_signals}
    return fig_data


def plot_raw_panel(ax, raw: dict, title: str | None = None) -> None:
    time = raw["time"]
    mean = raw["mean"]
    variance = raw["variance"]
    time, mean, variance = thin_for_plot(time[: len(mean)], mean, variance)
    ax.plot(time, mean, color=RAW_COLOR, linewidth=1.4)
    # ax.fill_between(
    #     time,
    #     mean - variance,
    #     mean + variance,
    #     color=RAW_COLOR,
    #     alpha=0.16,
    #     linewidth=0,
    # )
    if title:
        ax.set_title(title, pad=6)
    ax.set_ylabel("State")
    


def plot_model_panel(ax, signal: dict, model_name: str):
    color = MODEL_COLORS.get(model_name, "0.25")
    time = signal["time"]
   # ews = relative_uncertainty(signal["ews"])
    ews = signal["ews"]
    handle, = ax.plot(time[: len(ews)], ews, "o", color=color, markersize=2.0, linewidth=0, label=model_name)
    ax.set_ylabel("MPV")
    return handle


def add_transition_line(ax, x: float) -> None:
    ax.axvline(x=x, color=TRANSITION_COLOR, linestyle="--", linewidth=0.9, alpha=0.75, zorder=1)


def build_fig1(
    ews_root: Path = EWS_ROOT,
    source_root: Path = Source_ROOT,
    output_dir: Path = OUTPUT_ROOT,
    data_trend: str = "increase",
    graph: str = "barabasi_albert_30_0",
    models: tuple[str, ...] = FIG1_MODELS,
    force_recompute: bool = False,
) -> None:
    set_nature_style(font_size=10)
    output_dir = ensure_output(output_dir)
    fig_data = collect_fig1_data(
        ews_root=ews_root,
        source_root=source_root,
        data_trend=data_trend,
        graph=graph,
        models=models,
        force_recompute=force_recompute,
    )

    nrows = 1 + len(models)
    ncols = len(DATASETS)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(3.25 * ncols, 1.20 * nrows + 0.35),
        sharex="col",
        squeeze=False,
        gridspec_kw={"hspace": 0.08, "wspace": 0.24},
    )

    legend_handles = {}
    for col, dataset_type in enumerate(DATASETS):
        dataset_data = fig_data[dataset_type]
        plot_raw_panel(axs[0, col], dataset_data["raw"], title=dynamics_title(dataset_type))
        for row, model_name in enumerate(models, start=1):
            handle = plot_model_panel(axs[row, col], dataset_data["models"][model_name], model_name)
            legend_handles.setdefault(model_name, handle)
        for ax in axs[:, col]:
            add_transition_line(ax, dataset_data["raw"]["transition_time"])
        axs[-1, col].set_xlabel("Time")

    for row in range(nrows - 1):
        for ax in axs[row, :]:
            ax.tick_params(labelbottom=False)
    for ax in axs.ravel():
        ax.tick_params(length=3, width=0.8)
        ax.margins(x=0)

    for col, label in enumerate(("a", "b", "c")):
        axs[0, col].text(
            -0.14,
            1.20,
            label,
            transform=axs[0, col].transAxes,
            fontsize=12.5,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.legend(
        [legend_handles[model_name] for model_name in models if model_name in legend_handles],
        [model_name for model_name in models if model_name in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.52, 1.01),
        ncol=len(models),
        fontsize=8.3,
        handlelength=1.0,
        columnspacing=1.2,
    )

    output_base = output_dir / f"fig1_{data_trend}_{graph}"
    fig.savefig(output_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
   # fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)


    parser.add_argument("--graph", default="barabasi_albert_30_0")
    parser.add_argument("--models", nargs="+", default=list(FIG1_MODELS))
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    build_fig1(
        ews_root=args.ews_root,
        source_root=args.source_root,
        output_dir=args.output_dir,
        data_trend=args.trend,
        graph=args.graph,
        models=tuple(args.models),
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
