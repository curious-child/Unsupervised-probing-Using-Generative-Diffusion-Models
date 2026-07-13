"""Generate MPV collapse summary tables for reviewer statistics.

This script summarizes model-predicted variance collapse from NsDiff-g/``_gx``
caches. Raw network states are used only to locate and validate transitions.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_and_analysis.diffusion_model_uncertainy import uncertainty_ews
from paper_figures.common import DATASETS, EWS_ROOT, OUTPUT_ROOT, Source_ROOT, TRENDS


TRANSITION_EDGE_FRACTION = 0.10
TRANSITION_WINDOW_FRACTION = 0.05
MIN_TRANSITION_WINDOW = 10
SIS_THRESHOLD = 1e-2
DEFAULT_EXCLUDE_ETA = ("0.5",)
BASELINE_FRACTION_RANGE = (0.01, 0.20)
TOPOLOGY_TYPES = ("BA", "ER", "SW")
TOPOLOGY_TYPE_NAMES = {
    "BA": "barabasi_albert",
    "ER": "erdos_renyi",
    "SW": "small-world",
}
METRICS = (
    "pre_transition_drop_percent",
    "global_drop_percent",
    "lead_time",
    "min_slope_before_transition",
    "min_slope_global",
    "slope_lead_time",
)
METRIC_DESCRIPTIONS = {
    "pre_transition_drop_percent": (
        "mpv_collapse_pre_transition_drop_percent.csv",
        "MPV drop before the raw-state transition, relative to the early pre-transition baseline (%).",
    ),
    "global_drop_percent": (
        "mpv_collapse_global_drop_percent.csv",
        "Largest MPV drop over the whole trajectory, relative to the early pre-transition baseline (%).",
    ),
    "lead_time": (
        "mpv_collapse_lead_time.csv",
        "Time difference between the raw-state transition and the global MPV minimum; positive values mean early warning.",
    ),
    "min_slope_before_transition": (
        "mpv_collapse_min_slope_before_transition.csv",
        "Minimum trailing MPV local slope before the raw-state transition.",
    ),
    "min_slope_global": (
        "mpv_collapse_min_slope_global.csv",
        "Minimum trailing MPV local slope over the whole trajectory.",
    ),
    "slope_lead_time": (
        "mpv_collapse_slope_lead_time.csv",
        "Time difference between the raw-state transition and the global minimum trailing MPV local slope.",
    ),
}
COUNT_TABLE = "mpv_collapse_n_valid.csv"
SAMPLE_FIELDS = (
    "dynamic_type",
    "topology_type",
    "topology",
    "trend",
    "data_file",
    "transition_time",
    "is_transition",
    "skip_reason",
    "baseline_mpv",
    "min_mpv_before_transition",
    "min_mpv_global",
    "pre_transition_drop_percent",
    "global_drop_percent",
    "global_min_mpv_time",
    "lead_time",
    "min_slope_before_transition",
    "min_slope_before_transition_time",
    "min_slope_global",
    "min_slope_global_time",
    "slope_lead_time",
    "mpv_cache_path",
    "check_plot_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize NsDiff-g MPV collapse around raw-state transitions."
    )
    parser.add_argument("--source-root", type=Path, default=Source_ROOT)
    parser.add_argument("--ews-root", type=Path, default=EWS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model-name", default="NsDiff")
    parser.add_argument("--dynamic", choices=DATASETS, action="append")
    parser.add_argument("--trend", choices=TRENDS, action="append")
    parser.add_argument("--topology", action="append", help="Topology folder name to include; repeatable.")
    parser.add_argument("--topology-type", choices=TOPOLOGY_TYPES, action="append")
    parser.add_argument("--eta", action="append", help="Only include these eta values; repeatable.")
    parser.add_argument("--exclude-eta", action="append", default=list(DEFAULT_EXCLUDE_ETA))
    parser.add_argument("--pred-dim", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Scan inputs and transitions without model calls.")
    parser.add_argument("--plot-checks", action="store_true", help="Save per-sample diagnostic plots for raw state and MPV metrics.")
    parser.add_argument("--check-plot-dir", type=Path, default=None)
    parser.add_argument("--slope-window-ratio", type=float, default=0.05)
    parser.add_argument("--plot-summary-relations", action="store_true")
    parser.add_argument("--samples-csv", default="mpv_collapse_samples.csv")
    parser.add_argument("--summary-csv", default="mpv_collapse_summary.csv")
    parser.add_argument("--summary-md", default="mpv_collapse_summary.md")
    return parser.parse_args()


def as_float_or_nan(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def parse_eta(path: Path) -> str:
    match = re.search(r"eta([0-9.eE+-]+)", path.name)
    return match.group(1) if match else ""


def parse_trend(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_increase"):
        return "increase"
    if stem.endswith("_decrease"):
        return "decrease"
    return ""


def topology_type_from_name(topology: str) -> str:
    name = topology.lower()
    if name.startswith("barabasi_albert"):
        return "BA"
    if name.startswith("erdos_renyi"):
        return "ER"
    if name.startswith("small-world") or name.startswith("small_world"):
        return "SW"
    return ""


def iter_data_files(source_root: Path, dynamics, trends, topologies, topology_types, include_eta, exclude_eta):
    include_eta = set(include_eta or [])
    exclude_eta = set(exclude_eta or [])
    topology_set = set(topologies or [])
    topology_type_set = set(topology_types or [])
    for dynamic_type in dynamics:
        dynamic_root = source_root / f"spdata_sde_{dynamic_type}"
        if not dynamic_root.exists():
            continue
        for topology_dir in sorted(p for p in dynamic_root.iterdir() if p.is_dir()):
            topology_type = topology_type_from_name(topology_dir.name)
            if not topology_type:
                continue
            if topology_set and topology_dir.name not in topology_set:
                continue
            if topology_type_set and topology_type not in topology_type_set:
                continue
            for data_file in sorted(topology_dir.glob("*.pt")):
                trend = parse_trend(data_file)
                eta = parse_eta(data_file)
                if trend not in trends:
                    continue
                if include_eta and eta not in include_eta:
                    continue
                if eta in exclude_eta:
                    continue
                yield dynamic_type, topology_type, topology_dir.name, trend, eta, data_file


def load_raw_state(data_file: Path):
    record = torch.load(data_file, map_location="cpu")
    if not isinstance(record, dict) or "ys_dynamic" not in record or "ts_dynamic" not in record:
        raise ValueError(f"{data_file} must contain ys_dynamic and ts_dynamic.")
    ys = torch.as_tensor(record["ys_dynamic"], dtype=torch.float32)
    ts = torch.as_tensor(record["ts_dynamic"], dtype=torch.float32)
    if ys.ndim != 2:
        raise ValueError(f"{data_file} ys_dynamic must have shape [T, Node], got {tuple(ys.shape)}.")
    if ts.ndim != 1 or ts.numel() != ys.shape[0]:
        raise ValueError(f"{data_file} ts_dynamic must have shape [T].")
    return ts.cpu().numpy(), ys.mean(dim=1).cpu().numpy()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    window = max(1, min(int(window), len(values)))
    if window == 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def locate_transition(time: np.ndarray, state_mean: np.ndarray, dynamic_type: str, trend: str) -> dict:
    n = len(state_mean)
    if n < 2 * MIN_TRANSITION_WINDOW + 1:
        return {"is_transition": False, "skip_reason": "too_short"}

    if dynamic_type == "SIS":
        window_size = min(10, n)
        rolling_mean = np.array([np.mean(state_mean[i : i + window_size]) for i in range(n - window_size)])
        if trend == "increase":
            candidates = np.argwhere(rolling_mean > SIS_THRESHOLD).flatten()
        else:
            candidates = np.argwhere(rolling_mean < SIS_THRESHOLD).flatten()
        idx = int(candidates[0]) if len(candidates) else int(np.argmax(np.abs(np.diff(state_mean))))
        method = "sis_threshold" if len(candidates) else "sis_max_diff_fallback"
    else:
        window_size = min(10, n - 1)
        score = np.abs(state_mean[window_size:] - state_mean[:-window_size]) / window_size
        if len(score) == 0:
            return {"is_transition": False, "skip_reason": "too_short_for_change_score"}
        idx = int(np.argmax(score))
        method = "max_window_change"

    window = max(MIN_TRANSITION_WINDOW, min(n // 20, 100))
    before = state_mean[max(0, idx - window) : idx]
    after = state_mean[idx : min(n, idx + window)]
    pre_mean = float(np.nanmean(before)) if len(before) else np.nan
    post_mean = float(np.nanmean(after)) if len(after) else np.nan
    state_change = abs(post_mean - pre_mean) if np.isfinite(pre_mean) and np.isfinite(post_mean) else np.nan
    state_range = float(np.nanmax(state_mean) - np.nanmin(state_mean))
    change_ratio = state_change / state_range if state_range > 0 and np.isfinite(state_change) else np.nan

    return {
        "is_transition": True,
        "skip_reason": "",
        "transition_time": float(time[idx]),
        "transition_index": idx,
        "transition_method": method,
        "state_change": state_change,
        "state_change_ratio": change_ratio,
    }


def compute_local_slopes(time_points, values, window_points):
    time_points = np.asarray(time_points, dtype=float)
    values = np.asarray(values, dtype=float)
    window_points = max(3, min(int(window_points), len(values)))
    if len(values) < window_points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    slope_times = []
    slopes = []
    for end in range(window_points, len(values) + 1):
        start = end - window_points
        x = time_points[start:end]
        y = values[start:end]
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 3:
            continue
        x = x[finite]
        y = y[finite]
        x_centered = x - np.mean(x)
        denom = np.sum(x_centered * x_centered)
        if denom <= 0:
            continue
        slope = float(np.sum(x_centered * (y - np.mean(y))) / denom)
        slope_times.append(float(time_points[end - 1]))
        slopes.append(slope)
    return np.asarray(slope_times, dtype=float), np.asarray(slopes, dtype=float)


def slope_metrics_from_mpv(slope_times, slopes, transition_time):
    slope_times = np.asarray(slope_times, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    finite = np.isfinite(slope_times) & np.isfinite(slopes)
    slope_times = slope_times[finite]
    slopes = slopes[finite]
    if len(slopes) < 1:
        return {}

    pre_indices = np.flatnonzero(slope_times < transition_time)
    if len(pre_indices):
        min_pre_idx = int(pre_indices[int(np.nanargmin(slopes[pre_indices]))])
        min_pre = float(slopes[min_pre_idx])
        min_pre_time = float(slope_times[min_pre_idx])
    else:
        min_pre = np.nan
        min_pre_time = np.nan

    min_global_idx = int(np.nanargmin(slopes))
    min_global = float(slopes[min_global_idx])
    min_global_time = float(slope_times[min_global_idx])
    return {
        "min_slope_before_transition": min_pre,
        "min_slope_before_transition_time": min_pre_time,
        "min_slope_global": min_global,
        "min_slope_global_time": min_global_time,
        "slope_lead_time": float(transition_time - min_global_time),
    }


def metric_row_from_mpv(time_points, mpv_values, transition_time, slope_window_points=None) -> dict:
    time_points = np.asarray(time_points, dtype=float)
    mpv_values = np.asarray(mpv_values, dtype=float)
    finite = np.isfinite(time_points) & np.isfinite(mpv_values)
    time_points = time_points[finite]
    mpv_values = mpv_values[finite]
    if len(mpv_values) < 3:
        return {"skip_reason": "too_few_mpv_points"}

    pre_mask = time_points < transition_time
    pre_indices = np.flatnonzero(pre_mask)
    if len(pre_indices) < 3:
        return {"skip_reason": "too_few_pre_transition_mpv_points"}

    start = int(np.floor(len(pre_indices) * BASELINE_FRACTION_RANGE[0]))
    end = int(np.ceil(len(pre_indices) * BASELINE_FRACTION_RANGE[1]))
    end = max(start + 1, min(end, len(pre_indices)))
    baseline_values = mpv_values[pre_indices[start:end]]
    baseline_mpv = float(np.nanmean(baseline_values))
    if not np.isfinite(baseline_mpv) or baseline_mpv <= 0:
        return {"skip_reason": "invalid_baseline_mpv"}

    pre_values = mpv_values[pre_indices]
    min_pre_idx_local = int(np.nanargmin(pre_values))
    min_pre_idx = int(pre_indices[min_pre_idx_local])
    min_global_idx = int(np.nanargmin(mpv_values))
    min_pre = float(mpv_values[min_pre_idx])
    min_global = float(mpv_values[min_global_idx])
    min_pre_time = float(time_points[min_pre_idx])
    global_min_time = float(time_points[min_global_idx])
    metrics = {
        "baseline_mpv": baseline_mpv,
        "baseline_start_time": float(time_points[pre_indices[start]]),
        "baseline_end_time": float(time_points[pre_indices[end - 1]]),
        "min_mpv_before_transition": min_pre,
        "min_mpv_before_transition_time": min_pre_time,
        "min_mpv_global": min_global,
        "pre_transition_drop_percent": 100.0 * (baseline_mpv - min_pre) / baseline_mpv,
        "global_drop_percent": 100.0 * (baseline_mpv - min_global) / baseline_mpv,
        "global_min_mpv_time": global_min_time,
        "lead_time": float(transition_time - global_min_time),
        "skip_reason": "",
    }
    if slope_window_points is not None:
        slope_times, slopes = compute_local_slopes(time_points, mpv_values, slope_window_points)
        metrics.update(slope_metrics_from_mpv(slope_times, slopes, transition_time))
        metrics["slope_times"] = slope_times
        metrics["slopes"] = slopes
    return metrics


def empty_sample_row(dynamic_type, topology_type, topology, trend, eta, data_file, reason) -> dict:
    row = {field: "" for field in SAMPLE_FIELDS}
    row.update(
        {
            "dynamic_type": dynamic_type,
            "topology_type": topology_type,
            "topology": topology,
            "trend": trend,
            "data_file": str(data_file),
            "is_transition": False,
            "skip_reason": reason,
        }
    )
    return row


def model_dir_for(ews_root: Path, model_name: str, dynamic_type: str) -> Path:
    return ews_root / "model_compare" / model_name / dynamic_type


def statistics_cache_dir(ews_root: Path, dynamic_type: str, topology: str) -> Path:
    return ews_root / "statistics_dataset" / dynamic_type / topology


def check_plot_path(args, dynamic_type: str, topology_type: str, topology: str, data_file: Path) -> Path:
    root = args.check_plot_dir or (args.output_dir / "mpv_collapse_checks")
    return root / dynamic_type / topology_type / topology / f"{data_file.stem}_mpv_check.png"


def save_check_plot(path: Path, time, state_mean, mpv_time, mpv_values, transition, metrics, title: str) -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    time = np.asarray(time, dtype=float)
    state_mean = np.asarray(state_mean, dtype=float)
    mpv_time = np.asarray(mpv_time, dtype=float)
    mpv_values = np.asarray(mpv_values, dtype=float)
    transition_time = float(transition["transition_time"])

    slope_times = np.asarray(metrics.get("slope_times", []), dtype=float)
    slopes = np.asarray(metrics.get("slopes", []), dtype=float)

    fig, axs = plt.subplots(3, 1, figsize=(7.0, 5.4), sharex=True, gridspec_kw={"hspace": 0.12})
    axs[0].plot(time, state_mean, color="#0F4D92", linewidth=1.0)
    axs[0].axvline(transition_time, color="#B64342", linestyle="--", linewidth=0.9, label="Transition")
    axs[0].set_ylabel("State")
    axs[0].set_title(title, fontsize=8.5)
    axs[0].legend(loc="best", frameon=False, fontsize=6.8)

    axs[1].plot(mpv_time, mpv_values, color="#B64342", linewidth=1.0)
    axs[1].axvline(transition_time, color="#B64342", linestyle="--", linewidth=0.9, label="Transition")
    axs[1].axhline(float(metrics["baseline_mpv"]), color="#4D4D4D", linestyle=":", linewidth=0.9, label="Baseline MPV")
    axs[1].axvspan(
        float(metrics["baseline_start_time"]),
        float(metrics["baseline_end_time"]),
        color="#D8D8D8",
        alpha=0.25,
        linewidth=0,
        label="Baseline Window",
    )
    axs[1].scatter(
        [float(metrics["min_mpv_before_transition_time"])],
        [float(metrics["min_mpv_before_transition"])],
        color="#E28E2C",
        s=18,
        zorder=3,
        label="Pre-transition Min",
    )
    axs[1].scatter(
        [float(metrics["global_min_mpv_time"])],
        [float(metrics["min_mpv_global"])],
        color="#42949E",
        s=18,
        zorder=3,
        label="Global Min",
    )
    axs[1].set_ylabel("MPV")
    axs[1].legend(loc="best", frameon=False, fontsize=6.8, ncol=2)

    axs[2].plot(slope_times, slopes, color="#9A4D8E", linewidth=1.0)
    axs[2].axhline(0, color="#767676", linestyle=":", linewidth=0.8)
    axs[2].axvline(transition_time, color="#B64342", linestyle="--", linewidth=0.9, label="Transition")
    if np.isfinite(as_float_or_nan(metrics.get("min_slope_before_transition_time"))):
        axs[2].scatter(
            [float(metrics["min_slope_before_transition_time"])],
            [float(metrics["min_slope_before_transition"])],
            color="#E28E2C",
            s=18,
            zorder=3,
            label="Pre-transition Min Slope",
        )
    if np.isfinite(as_float_or_nan(metrics.get("min_slope_global_time"))):
        axs[2].scatter(
            [float(metrics["min_slope_global_time"])],
            [float(metrics["min_slope_global"])],
            color="#42949E",
            s=18,
            zorder=3,
            label="Global Min Slope",
        )
    axs[2].set_ylabel("MPV Slope")
    axs[2].set_xlabel("Time")
    axs[2].legend(loc="best", frameon=False, fontsize=6.8, ncol=2)

    for ax in axs:
        ax.set_xlim(float(np.nanmin(time)), float(np.nanmax(time)))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=6.8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze_file(args, dynamic_type, topology_type, topology, trend, eta, data_file) -> dict:
    try:
        time, state_mean = load_raw_state(data_file)
        transition = locate_transition(time, state_mean, dynamic_type=dynamic_type, trend=trend)
    except Exception as exc:
        return empty_sample_row(dynamic_type, topology_type, topology, trend, eta, data_file, f"raw_error:{exc}")

    row = empty_sample_row(dynamic_type, topology_type, topology, trend, eta, data_file, transition.get("skip_reason", ""))
    row.update(
        {
            "transition_time": transition.get("transition_time", ""),
            "is_transition": bool(transition.get("is_transition", False)),
        }
    )
    if not transition.get("is_transition", False):
        return row

    model_dir = model_dir_for(args.ews_root, args.model_name, dynamic_type)
    if not model_dir.exists():
        row.update({"is_transition": False, "skip_reason": f"missing_model_dir:{model_dir}"})
        return row

    if args.dry_run:
        row.update({"skip_reason": "dry_run_no_model_call"})
        return row

    try:
        gx_cache_dir = statistics_cache_dir(args.ews_root, dynamic_type, topology)
        result = uncertainty_ews(
            model_save_file=model_dir,
            data_file=data_file,
            dynamic_type=dynamic_type,
            cache_path=model_dir,
            nsdiff_g_path=gx_cache_dir,
            uncertainty_method="gx",
            pred_dim=args.pred_dim,
            force_recompute=args.force_recompute,
        )
        slope_window_points = max(3, int(round(float(result["windows"]) * args.slope_window_ratio)))
        metrics = metric_row_from_mpv(
            result["time_points"],
            result["ews"],
            transition_time=float(transition["transition_time"]),
            slope_window_points=slope_window_points,
        )
    except Exception as exc:
        row.update({"is_transition": False, "skip_reason": f"mpv_error:{exc}"})
        return row

    if metrics.get("skip_reason"):
        row.update({"is_transition": False, "skip_reason": metrics["skip_reason"]})
        return row

    slope_times = metrics.pop("slope_times", np.asarray([], dtype=float))
    slopes = metrics.pop("slopes", np.asarray([], dtype=float))
    plot_metrics = dict(metrics)
    plot_metrics["slope_times"] = slope_times
    plot_metrics["slopes"] = slopes
    row["mpv_time"] = np.asarray(result["time_points"], dtype=float)
    row["mpv_values"] = np.asarray(result["ews"], dtype=float)

    if args.plot_checks:
        plot_path = check_plot_path(args, dynamic_type, topology_type, topology, data_file)
        save_check_plot(
            plot_path,
            time=time,
            state_mean=state_mean,
            mpv_time=result["time_points"],
            mpv_values=result["ews"],
            transition=transition,
            metrics=plot_metrics,
            title=f"{dynamic_type} | {topology_type} | {topology} | {trend} | {data_file.name}",
        )
        row["check_plot_path"] = str(plot_path)

    row.update(metrics)
    row.update({"skip_reason": "", "mpv_cache_path": result.get("cache_path", "")})
    return row


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize_metric(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {
            "n_valid": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }
    return {
        "n_valid": int(len(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.nanmedian(arr)),
        "q25": float(np.nanpercentile(arr, 25)),
        "q75": float(np.nanpercentile(arr, 75)),
    }


def build_summary_rows(sample_rows: list[dict]) -> list[dict]:
    valid_rows = [row for row in sample_rows if str(row.get("skip_reason", "")) == ""]
    summary_rows = []
    grouped = defaultdict(list)
    for row in valid_rows:
        key = (row.get("dynamic_type", ""), row.get("topology_type", ""), row.get("trend", ""))
        grouped[key].append(row)
    for (dynamic_type, topology_type, trend), rows in sorted(grouped.items()):
        for metric in METRICS:
            stats = summarize_metric([as_float_or_nan(row.get(metric)) for row in rows])
            summary_rows.append(
                {
                    "dynamic_type": dynamic_type,
                    "topology_type": topology_type,
                    "trend": trend,
                    "metric": metric,
                    **stats,
                }
            )
    return summary_rows


def format_value(value, digits=2) -> str:
    value = as_float_or_nan(value)
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def topology_trend_pairs() -> list[tuple[str, str]]:
    return [(topology_type, trend) for topology_type in TOPOLOGY_TYPES for trend in TRENDS]


def cell_text(stats: dict, metric: str) -> str:
    if int(stats.get("n_valid", 0)) == 0:
        return ""
    suffix = "%" if metric.endswith("percent") else ""
    return "{mean}{suffix} +/- {std}{suffix}".format(
        mean=format_value(stats["mean"]),
        std=format_value(stats["std"]),
        suffix=suffix,
    )


def build_metric_matrix(summary_rows: list[dict], metric: str) -> tuple[list[str], list[dict]]:
    by_key = {
        (row["dynamic_type"], row["topology_type"], row["trend"], row["metric"]): row
        for row in summary_rows
    }
    pairs = topology_trend_pairs()
    fields = ["dynamic_type"] + [f"{topology_type} | {trend}" for topology_type, trend in pairs]
    dynamics = [dynamic for dynamic in DATASETS if any(row["dynamic_type"] == dynamic for row in summary_rows)]
    rows = []
    for dynamic_type in dynamics:
        out = {"dynamic_type": dynamic_type}
        for topology_type, trend in pairs:
            key = (dynamic_type, topology_type, trend, metric)
            out[f"{topology_type} | {trend}"] = cell_text(by_key[key], metric) if key in by_key else ""
        rows.append(out)
    return fields, rows


def build_count_matrix(summary_rows: list[dict]) -> tuple[list[str], list[dict]]:
    metric = METRICS[0]
    by_key = {
        (row["dynamic_type"], row["topology_type"], row["trend"], row["metric"]): row
        for row in summary_rows
    }
    pairs = topology_trend_pairs()
    fields = ["dynamic_type"] + [f"{topology_type} | {trend}" for topology_type, trend in pairs]
    dynamics = [dynamic for dynamic in DATASETS if any(row["dynamic_type"] == dynamic for row in summary_rows)]
    rows = []
    for dynamic_type in dynamics:
        out = {"dynamic_type": dynamic_type}
        for topology_type, trend in pairs:
            key = (dynamic_type, topology_type, trend, metric)
            out[f"{topology_type} | {trend}"] = int(by_key[key]["n_valid"]) if key in by_key else ""
        rows.append(out)
    return fields, rows


def write_display_tables(output_dir: Path, summary_rows: list[dict]) -> list[Path]:
    written = []
    for metric, (filename, _description) in METRIC_DESCRIPTIONS.items():
        fields, rows = build_metric_matrix(summary_rows, metric)
        path = output_dir / filename
        write_csv(path, rows, fields)
        written.append(path)
    fields, rows = build_count_matrix(summary_rows)
    path = output_dir / COUNT_TABLE
    write_csv(path, rows, fields)
    written.append(path)
    return written


def sample_group_key(row):
    return row.get("dynamic_type", ""), row.get("trend", "")


def summarize_xy_records(records, x_key, y_key):
    grouped = defaultdict(list)
    for record in records:
        x = as_float_or_nan(record.get(x_key))
        y = as_float_or_nan(record.get(y_key))
        if np.isfinite(x) and np.isfinite(y):
            grouped[x].append(y)
    rows = []
    for x in sorted(grouped):
        values = np.asarray(grouped[x], dtype=float)
        rows.append(
            {
                x_key: x,
                "mean_lead_time": float(np.nanmean(values)),
                "std_lead_time": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_valid": int(len(values)),
            }
        )
    return rows


def threshold_records_from_runtime(runtime_records):
    max_drop_by_group = defaultdict(float)
    for record in runtime_records:
        if str(record.get("skip_reason", "")) != "":
            continue
        if "mpv_values" not in record:
            continue
        baseline = as_float_or_nan(record.get("baseline_mpv"))
        if not np.isfinite(baseline) or baseline <= 0:
            continue
        drops = (baseline - record["mpv_values"]) / baseline
        if len(drops) == 0 or not np.isfinite(drops).any():
            continue
        group = (record["dynamic_type"], record["trend"])
        max_drop_by_group[group] = max(max_drop_by_group[group], float(np.nanmax(drops)))

    thresholds_by_group = {}
    for group, max_drop in max_drop_by_group.items():
        if max_drop >= 0.1:
            thresholds_by_group[group] = np.linspace(0.1, max_drop, 8)

    records = []
    for record in runtime_records:
        if str(record.get("skip_reason", "")) != "":
            continue
        if "mpv_values" not in record or "mpv_time" not in record:
            continue
        group = (record["dynamic_type"], record["trend"])
        baseline = as_float_or_nan(record.get("baseline_mpv"))
        if not np.isfinite(baseline) or baseline <= 0:
            continue
        drops = (baseline - record["mpv_values"]) / baseline
        for threshold in thresholds_by_group.get(group, []):
            hits = np.flatnonzero(drops >= threshold)
            if len(hits) == 0:
                continue
            hit_time = float(record["mpv_time"][int(hits[0])])
            records.append(
                {
                    "dynamic_type": record["dynamic_type"],
                    "trend": record["trend"],
                    "threshold": float(threshold),
                    "lead_time": float(record["transition_time"] - hit_time),
                }
            )
    return records


def slope_relation_records(sample_rows):
    records = []
    for row in sample_rows:
        if str(row.get("skip_reason", "")) != "":
            continue
        slope = as_float_or_nan(row.get("min_slope_global"))
        lead_time = as_float_or_nan(row.get("slope_lead_time"))
        if np.isfinite(slope) and np.isfinite(lead_time):
            records.append(
                {
                    "dynamic_type": row.get("dynamic_type", ""),
                    "trend": row.get("trend", ""),
                    "min_slope_global": slope,
                    "lead_time": lead_time,
                }
            )
    return records


def plot_six_panel_relation(
    records,
    x_key,
    y_key,
    output_path,
    xlabel,
    ylabel,
):
    import string
    import numpy as np
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    # 使用局部样式，避免影响脚本中的其他图片
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    with mpl.rc_context(style):
        fig, axs = plt.subplots(
            2,
            3,
            figsize=(7.1, 4.6),

            # 每个子图的坐标范围完全独立
            sharex=False,
            sharey=False,

            gridspec_kw={
                "hspace": 0.36,
                "wspace": 0.32,
            },
        )

        # 第一行对应第一个 trend，第二行对应第二个 trend；
        # 图中不显示 trend 名称
        panel_order = [
            (dynamic, trend)
            for trend in TRENDS
            for dynamic in DATASETS
        ]

        panel_labels = string.ascii_lowercase[:6]

        for panel_index, (
            ax,
            (dynamic_type, trend),
        ) in enumerate(zip(axs.ravel(), panel_order)):

            panel_records = [
                record
                for record in records
                if record.get("dynamic_type") == dynamic_type
                and record.get("trend") == trend
            ]

            if panel_records:
                rows = summarize_xy_records(
                    panel_records,
                    x_key,
                    y_key,
                )

                x = np.asarray(
                    [record[x_key] for record in rows],
                    dtype=float,
                )

                y = np.asarray(
                    [record["mean_lead_time"] for record in rows],
                    dtype=float,
                )

                sd = np.asarray(
                    [record["std_lead_time"] for record in rows],
                    dtype=float,
                )

                # 去除无效值
                valid = (
                    np.isfinite(x)
                    & np.isfinite(y)
                    & np.isfinite(sd)
                )

                x = x[valid]
                y = y[valid]
                sd = sd[valid]

                # 确保折线按横坐标顺序连接
                order = np.argsort(x)
                x = x[order]
                y = y[order]
                sd = sd[order]

                if len(x) > 0:
                    ax.fill_between(
                        x,
                        y - sd,
                        y + sd,
                        color="#0F4D92",
                        alpha=0.16,
                        linewidth=0,
                        zorder=1,
                    )

                    ax.plot(
                        x,
                        y,
                        color="#0F4D92",
                        linewidth=1.1,
                        marker="o",
                        markersize=3,
                        markeredgewidth=0,
                        zorder=2,
                    )

                    ax.axhline(
                        0,
                        color="#767676",
                        linestyle=":",
                        linewidth=0.7,
                        zorder=0,
                    )

                    if x_key == "min_slope_global":
                        ax.ticklabel_format(
                            axis="x",
                            style="sci",
                            scilimits=(-2, 2),
                            useMathText=True,
                        )

                        ax.xaxis.get_offset_text().set_fontsize(6)

                    if len(x) == 1:
                        pad = max(
                            abs(float(x[0])) * 0.2,
                            1e-6,
                        )

                        ax.set_xlim(
                            float(x[0]) - pad,
                            float(x[0]) + pad,
                        )
                    else:
                        ax.margins(x=0.04)

            else:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="#767676",
                    fontsize=7,
                )

            # 只保留动力学类型作为标题，不显示 trend
            ax.set_title(
                str(dynamic_type),
                pad=4,
                fontweight="normal",
            )

            # 面板编号放在子图外部左上角
            ax.text(
                -0.15,
                1.10,
                panel_labels[panel_index],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.2,
                fontweight="bold",
                clip_on=False,
            )

            ax.set_xlabel(
                xlabel,
                fontsize=8,
                labelpad=3,
            )

            ax.set_ylabel(
                ylabel,
                fontsize=8,
                labelpad=3,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.tick_params(
                axis="both",
                which="major",
                labelsize=6.6,
                pad=2,
            )

            # 不加入额外网格
            ax.grid(False)

        # 控制整体边距，为外部面板编号留出空间
        fig.subplots_adjust(
            left=0.105,
            right=0.985,
            bottom=0.105,
            top=0.92,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )

        plt.close(fig)


def write_relation_csv(path, records, x_key):
    fields = ["dynamic_type", "trend", x_key, "mean_lead_time", "std_lead_time", "n_valid"]
    rows = []
    for dynamic_type in DATASETS:
        for trend in TRENDS:
            panel_records = [r for r in records if r.get("dynamic_type") == dynamic_type and r.get("trend") == trend]
            for row in summarize_xy_records(panel_records, x_key, "lead_time"):
                rows.append({"dynamic_type": dynamic_type, "trend": trend, **row})
    write_csv(path, rows, fields)


def write_markdown_summary(path: Path, summary_rows: list[dict], sample_rows: list[dict], display_paths: list[Path]) -> None:
    valid_count = sum(1 for row in sample_rows if str(row.get("skip_reason", "")) == "")
    total_count = len(sample_rows)
    lines = [
        "# MPV Collapse Summary",
        "",
        "Purpose: these tables support the reviewer-requested statistical summary of MPV collapse across network dynamics and topologies.",
        "Rows are dynamical systems; columns are topology classes (BA, ER, SW) and control-parameter trend. Noise levels are used only for data filtering and are not reported as table columns.",
        "Topology classes aggregate all matching graph instances: BA = barabasi_albert, ER = erdos_renyi, SW = small-world.",
        "",
        f"Total scanned samples: {total_count}",
        f"Valid transition samples with MPV metrics: {valid_count}",
        "",
        "## Manuscript-facing CSV tables",
        "",
        f"- `{COUNT_TABLE}`: number of valid transition-bearing samples used in each dynamical-system/topology-class/trend cell.",
    ]
    for metric, (filename, description) in METRIC_DESCRIPTIONS.items():
        lines.append(f"- `{filename}`: {description}")
    lines.extend(
        [
            "",
            "Each metric display cell is formatted as `mean +/- SD`; percentage signs denote MPV reduction percentages.",
            "",
            "## Source-data table",
            "",
            "`mpv_collapse_samples.csv` contains one row per scanned data file for audit and reproducibility. Rows with non-empty `skip_reason` are excluded from summary statistics.",
            "`mpv_collapse_summary.csv` contains the numeric long-format source table behind the display CSV files.",
            "",
            "Positive lead_time means the global MPV minimum precedes the raw-state transition time.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_fields() -> list[str]:
    return ["dynamic_type", "topology_type", "trend", "metric", "n_valid", "mean", "std", "median", "q25", "q75"]


def main() -> None:
    args = parse_args()
    dynamics = tuple(args.dynamic) if args.dynamic else DATASETS
    trends = tuple(args.trend) if args.trend else TRENDS
    files = list(
        iter_data_files(
            source_root=args.source_root,
            dynamics=dynamics,
            trends=trends,
            topologies=args.topology,
            topology_types=args.topology_type,
            include_eta=args.eta,
            exclude_eta=args.exclude_eta,
        )
    )
    if args.max_files is not None:
        files = files[: args.max_files]

    sample_rows = [analyze_file(args, *file_info) for file_info in files]
    summary_rows = build_summary_rows(sample_rows)

    output_dir = args.output_dir
    write_csv(output_dir / args.samples_csv, sample_rows, SAMPLE_FIELDS)
    write_csv(output_dir / args.summary_csv, summary_rows, summary_fields())
    display_paths = write_display_tables(output_dir, summary_rows)
    relation_paths = []
    if args.plot_summary_relations:
        drop_records = threshold_records_from_runtime(sample_rows)
        drop_csv = output_dir / "mpv_drop_threshold_lead_time.csv"
        drop_png = output_dir / "mpv_drop_threshold_lead_time.png"
        write_relation_csv(drop_csv, drop_records, "threshold")
        plot_six_panel_relation(
            drop_records,
            x_key="threshold",
            y_key="lead_time",
            output_path=drop_png,
            xlabel="MPV drop threshold",
            ylabel="Lead time",
        )
        relation_paths.extend([drop_csv, drop_png])

        slope_records = slope_relation_records(sample_rows)
        slope_csv = output_dir / "mpv_slope_lead_time.csv"
        slope_png = output_dir / "mpv_slope_lead_time.png"
        write_relation_csv(slope_csv, slope_records, "min_slope_global")
        plot_six_panel_relation(
            slope_records,
            x_key="min_slope_global",
            y_key="lead_time",
            output_path=slope_png,
            xlabel="Minimum MPV local slope",
            ylabel="Lead time",
        )
        relation_paths.extend([slope_csv, slope_png])
    write_markdown_summary(output_dir / args.summary_md, summary_rows, sample_rows, display_paths)

    print(f"scanned_files: {len(files)}")
    print(f"valid_rows: {sum(1 for row in sample_rows if str(row.get('skip_reason', '')) == '')}")
    print(f"samples_csv: {output_dir / args.samples_csv}")
    print(f"summary_csv: {output_dir / args.summary_csv}")
    print(f"summary_md: {output_dir / args.summary_md}")
    for path in display_paths:
        print(f"display_table: {path}")
    for path in relation_paths:
        print(f"relation_output: {path}")


if __name__ == "__main__":
    main()
