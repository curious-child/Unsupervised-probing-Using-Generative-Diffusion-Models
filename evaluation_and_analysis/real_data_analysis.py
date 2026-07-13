"""Build paper-ready real-data EWS subfigures.

This file is self-contained and intentionally does not import
``diffusion_real_data_grid.py``. It copies the small pieces needed for real-data
loading, classic EWS computation, cached NsDiff uncertainty parsing, optional
NsDiff re-computation, and Bury ML probability plotting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from ewstools import TimeSeries
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import load_diffusion_model  # noqa: E402


BURY_ML_PROB_COLUMNS = ["fold_prob", "hopf_prob", "branch_prob", "null_prob"]
VALID_SIGNALS = {"model_uncertainty", "model_trend", "ar1", "variance", "sample-entropy-1", "bury_ml"}
MODEL_KEY_DIRS = {
    "simulation": PROJECT_ROOT / "ews_results" / "real_data" / "model" / "simulation",
    "Real_Neuronal": PROJECT_ROOT / "ews_results" / "real_data" / "model" / "Real_Neuronal",
}


def as_float_tensor(value):
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def resolve_real_data_path(name, data_root):
    path = Path(name)
    if path.suffix == ".pt" and path.exists():
        return path
    matches = sorted(Path(data_root).rglob(f"{name}.pt"))
    if not matches:
        raise FileNotFoundError(f"Could not find real-data file for {name} under {data_root}")
    return matches[0]


def data_name_from_record(record, data_path):
    return str(record.get("name", data_path.stem))


def get_transition_time_from_record(record):
    ts = as_float_tensor(record.get("ts_dynamic", [])).flatten()
    if len(ts) == 0:
        return None
    if "transition_index" in record:
        idx = max(0, min(int(record["transition_index"]), len(ts) - 1))
        return float(ts[idx])
    if "transition_time" in record:
        return float(record["transition_time"])
    if "transition_age" in record:
        transition_age = float(record["transition_age"])
        nearest = torch.argmin(torch.abs(ts - transition_age))
        return float(ts[int(nearest)])
    return None


def ensure_min_time_points_linear_interp(record, min_sampled_points=200, sampling_interval=1):
    """Interpolate a record only when re-running a model needs more input points."""
    ys_dynamic = as_float_tensor(record["ys_dynamic"])
    ts_dynamic = as_float_tensor(record["ts_dynamic"]).flatten()
    if ys_dynamic.ndim == 1:
        ys_dynamic = ys_dynamic.unsqueeze(-1)

    sampling_interval = max(1, int(sampling_interval))
    if len(ts_dynamic[::sampling_interval]) >= int(min_sampled_points):
        record["ys_dynamic"] = ys_dynamic
        record["ts_dynamic"] = ts_dynamic
        return record, False
    if len(ts_dynamic) < 2:
        raise ValueError("At least two time points are required for interpolation.")

    old_ts = ts_dynamic.numpy().astype(float)
    old_ys = ys_dynamic.numpy().astype(float)
    order = np.argsort(old_ts)
    old_ts = old_ts[order]
    old_ys = old_ys[order]
    unique_ts, unique_idx = np.unique(old_ts, return_index=True)
    unique_ys = old_ys[unique_idx]

    target_len = max(len(unique_ts), (int(min_sampled_points) - 1) * sampling_interval + 1)
    new_ts = np.linspace(unique_ts[0], unique_ts[-1], target_len, dtype=np.float32)
    new_ys = np.stack(
        [np.interp(new_ts, unique_ts, unique_ys[:, dim]) for dim in range(unique_ys.shape[1])],
        axis=1,
    ).astype(np.float32)

    old_transition_time = get_transition_time_from_record(record)
    record["ys_dynamic"] = torch.tensor(new_ys, dtype=torch.float32)
    record["ts_dynamic"] = torch.tensor(new_ts, dtype=torch.float32)
    record["num_time_points"] = int(target_len)
    if old_transition_time is not None:
        new_idx = int(np.argmin(np.abs(new_ts - old_transition_time)))
        record["transition_index"] = new_idx
        record["transition_time"] = float(new_ts[new_idx])
    if "tp_values" in record:
        record["tp_values"] = torch.zeros(target_len, dtype=torch.float32)
    return record, True


def detrend_1d_series(data, method="Gaussian", span=0.2, bandwidth=0.2):
    method_key = str(method).lower()
    values = np.asarray(data, dtype=float)
    if method_key in {"none", "raw"}:
        return torch.tensor(values, dtype=torch.float32)
    if np.isnan(values).any():
        good = np.flatnonzero(~np.isnan(values))
        values = np.interp(np.arange(len(values)), good, values[good])
    ts = TimeSeries(values)
    if method_key == "lowess":
        ts.detrend(method="Lowess", span=span)
    elif method_key == "gaussian":
        ts.detrend(method="Gaussian", bandwidth=bandwidth)
    else:
        raise ValueError(f"Unsupported detrend method: {method}")
    return torch.tensor(ts.state["residuals"].to_numpy(dtype=float), dtype=torch.float32)


def prepare_model_input_series(ys_dynamic, model_input, detrend_method, detrend_span, detrend_bandwidth):
    ys_dynamic = as_float_tensor(ys_dynamic)
    if ys_dynamic.ndim == 1:
        ys_dynamic = ys_dynamic.unsqueeze(-1)
    if model_input == "raw":
        return ys_dynamic
    if model_input != "detrended":
        raise ValueError(f"Unsupported model input: {model_input}")
    columns = [
        detrend_1d_series(
            ys_dynamic[:, dim].numpy(),
            method=detrend_method,
            span=detrend_span,
            bandwidth=detrend_bandwidth,
        )
        for dim in range(ys_dynamic.shape[1])
    ]
    return torch.stack(columns, dim=1)


class EWSModelEval(TimeSeries):
    def compute_indicator(self, indicator, **kwargs):
        rolling_window = kwargs.get("rolling_window", 0.5)
        if indicator == "variance":
            self.compute_var(rolling_window=rolling_window)
            return self.ews["variance"]
        if indicator == "ar1":
            self.compute_auto(lag=1, rolling_window=rolling_window)
            return self.ews["ac1"]
        if indicator == "sample-entropy-1":
            return compute_sample_entropy_1(self, rolling_window=rolling_window)
        raise ValueError(f"Unsupported indicator: {indicator}")


def compute_sample_entropy_1(ts: TimeSeries, rolling_window=0.5):
    """Compute ewstools sample entropy and return the m=1 signal."""
    ts.compute_entropy(rolling_window=rolling_window, method="sample")
    if "sample-entropy-1" in ts.ews:
        return ts.ews["sample-entropy-1"]
    sample_cols = [col for col in ts.ews.columns if str(col).startswith("sample-entropy")]
    if not sample_cols:
        raise ValueError("ewstools did not produce sample entropy columns.")
    return ts.ews[sample_cols[min(1, len(sample_cols) - 1)]]


def compute_classic_ews(times, values, method="Gaussian", span=0.2, bandwidth=0.2, rolling_window=0.5):
    series = pd.Series(np.asarray(values, dtype=float), index=np.asarray(times, dtype=float))
    ts = EWSModelEval(series)
    if str(method).lower() == "lowess":
        ts.detrend(method="Lowess", span=span)
    elif str(method).lower() == "gaussian":
        ts.detrend(method="Gaussian", bandwidth=bandwidth)
    elif str(method).lower() not in {"none", "raw"}:
        raise ValueError(f"Unsupported EWS detrend method: {method}")
    variance = ts.compute_indicator("variance", rolling_window=rolling_window)
    ar1 = ts.compute_indicator("ar1", rolling_window=rolling_window)
    sample_entropy = ts.compute_indicator("sample-entropy-1", rolling_window=rolling_window)
    return {
        "variance": pd.Series(variance.to_numpy(dtype=float), index=variance.index.to_numpy(dtype=float)),
        "ar1": pd.Series(ar1.to_numpy(dtype=float), index=ar1.index.to_numpy(dtype=float)),
        "sample-entropy-1": pd.Series(sample_entropy.to_numpy(dtype=float), index=sample_entropy.index.to_numpy(dtype=float)),
    }


def torch_data_preprocessing(time_data, sampling_t, return_numpy=False):
    sampling_t_min = 0.1
    sampling_interval = int(sampling_t / sampling_t_min) if sampling_t > sampling_t_min else 1
    if return_numpy:
        return time_data[::sampling_interval].detach().cpu().numpy()
    return time_data[:, ::sampling_interval, :]


def compute_variance_trend(time_points, variance_values, trend_window=40, min_points=5, normalize_time=True):
    time_points = np.asarray(time_points, dtype=float)
    variance_values = np.asarray(variance_values, dtype=float)
    if len(time_points) != len(variance_values):
        raise ValueError("time_points and variance_values must have the same length.")
    trend_window = min(int(trend_window), len(variance_values))
    if trend_window < min_points:
        return np.array([]), np.array([])
    trend_times, trend_values = [], []
    for end_idx in range(trend_window, len(variance_values) + 1):
        start_idx = end_idx - trend_window
        t_window = time_points[start_idx:end_idx]
        v_window = variance_values[start_idx:end_idx]
        valid = np.isfinite(t_window) & np.isfinite(v_window)
        if valid.sum() < min_points:
            continue
        t_valid = t_window[valid]
        v_valid = v_window[valid]
        if normalize_time:
            t_span = t_valid.max() - t_valid.min()
            if t_span == 0:
                continue
            t_valid = (t_valid - t_valid.min()) / t_span
        slope, _ = np.polyfit(t_valid, v_valid, deg=1)
        trend_times.append(time_points[end_idx - 1])
        trend_values.append(slope)
    return np.asarray(trend_times), np.asarray(trend_values)


def model_cache_path(data_name, real_data_result_root):
    return Path(real_data_result_root) / "data" / data_name / "model_uncertainty.pt"


def parse_model_uncertainty_cache(cache_path, pred_dim):
    obj = torch.load(cache_path, map_location="cpu")
    if isinstance(obj, dict):
        if "model_uncertainty" in obj and "time" in obj:
            return np.asarray(obj["time"], dtype=float), np.asarray(obj["model_uncertainty"], dtype=float)
        if "values" in obj and "time" in obj:
            return np.asarray(obj["time"], dtype=float), np.asarray(obj["values"], dtype=float)
        if "data_save_list" in obj:
            obj = obj["data_save_list"]
        else:
            raise ValueError(f"Unsupported model uncertainty dict keys in {cache_path}: {list(obj.keys())}")
    if not isinstance(obj, list):
        raise ValueError(f"Expected list[Tensor] in {cache_path}, got {type(obj)}")
    values = []
    for gx in obj:
        gx = gx.detach().cpu()
        ews = gx.mean(dim=-1).numpy()
        if pred_dim >= len(ews):
            raise ValueError(f"pred_dim={pred_dim} invalid for cached gx shape {tuple(gx.shape)}")
        values.append(float(ews[pred_dim]))
    return None, np.asarray(values, dtype=float)


def infer_model_times(time_data, num_values, sample_window_step=1, sampling_t=0.1, model_window=None):
    sampling_t_min = 0.1
    sampling_interval = int(sampling_t / sampling_t_min) if sampling_t > sampling_t_min else 1
    sampled_time = as_float_tensor(time_data).flatten()[::sampling_interval].numpy()
    if model_window is None:
        model_window = len(sampled_time) - (int(num_values) - 1) * int(sample_window_step)
    model_window = int(model_window)
    if model_window < 1:
        raise ValueError(
            f"Cannot infer model window from sampled length={len(sampled_time)}, "
            f"num_values={num_values}, sample_window_step={sample_window_step}."
        )
    return sampled_time[model_window - 1 :: int(sample_window_step)][:num_values]


def resolve_model_dir(model_key=None, model_dir=None):
    if model_dir is not None:
        return Path(model_dir)
    if model_key is None:
        return None
    if model_key not in MODEL_KEY_DIRS:
        raise ValueError(f"model_key must be one of {sorted(MODEL_KEY_DIRS)}")
    return MODEL_KEY_DIRS[model_key]


def load_model_config(model_dir):
    model_dir = Path(model_dir)
    yaml_path = model_dir / "model_trained.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing model config: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle), yaml_path


def run_model_uncertainty(record, data_name, cache_path, model_dir, args):
    model_config, _ = load_model_config(model_dir)
    windows = int(model_config["dataset"]["windows"])
    sampling_t = float(model_config["dataset"].get("sampling_t", args.sampling_t))
    record, interpolated = ensure_min_time_points_linear_interp(
        record,
        min_sampled_points=windows,
        sampling_interval=max(1, int(sampling_t / 0.1)) if sampling_t > 0.1 else 1,
    )
    if interpolated:
        print(f"{data_name}: interpolated to {record['num_time_points']} points for model window={windows}")

    model_input = prepare_model_input_series(
        record["ys_dynamic"],
        model_input=args.model_input,
        detrend_method=args.model_detrend_method,
        detrend_span=args.model_detrend_span,
        detrend_bandwidth=args.model_detrend_bandwidth,
    )
    torch_model_time_series = model_input.t().unsqueeze(-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_params = {"parallel_sample": args.parallel_sample, "n_z_samples": args.n_z_samples}
    model_path = Path(model_dir) / "model_trained"
    model, _ = load_diffusion_model(
        str(model_path),
        device=device,
        infer_para=infer_params,
        dataparallel=False,
        train_model_select=model_config["train"].get("train_model_select"),
    )
    model.eval()

    sampled_series = torch_data_preprocessing(torch_model_time_series, sampling_t=sampling_t)
    sampled_time = torch_data_preprocessing(as_float_tensor(record["ts_dynamic"]), sampling_t=sampling_t, return_numpy=True)
    timeseries_data = sampled_series.unfold(1, windows, args.sample_window_step).permute(0, 1, 3, 2)
    model_times = sampled_time[windows - 1 :: args.sample_window_step]
    data_save_list = []
    values = []
    with torch.no_grad():
        for time_series in tqdm(timeseries_data.unbind(1), desc=f"{data_name} model uncertainty"):
            if getattr(model, "scaler", None) == "StandardScaler":
                time_series = model.scaler_transform(time_series.to(device))
            else:
                time_series = time_series.to(device)
            gx = model.cond_pred_model_g(time_series).squeeze(-1)
            data_save_list.append(gx.detach().cpu())
            ews = gx.mean(dim=-1).detach().cpu().numpy()
            values.append(float(ews[args.pred_dim]))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_save_list, cache_path)
    return model_times[: len(values)], np.asarray(values, dtype=float)


def load_or_run_model_uncertainty(record, data_name, args):
    cache_path = model_cache_path(data_name, args.real_data_result_root)
    model_dir = resolve_model_dir(args.model_key, args.model_dir)
    if cache_path.exists():
        cached_time, values = parse_model_uncertainty_cache(cache_path, args.pred_dim)
        if cached_time is None:
            model_window = None
            if model_dir is not None:
                model_config, _ = load_model_config(model_dir)
                model_window = int(model_config["dataset"]["windows"])
            cached_time = infer_model_times(
                record["ts_dynamic"],
                len(values),
                sample_window_step=args.sample_window_step,
                sampling_t=args.sampling_t,
                model_window=model_window,
            )
        return cached_time, values, cache_path
    if model_dir is None:
        raise FileNotFoundError(
            f"Missing {cache_path}. Specify --model-key simulation|Real_Neuronal or --model-dir to regenerate it."
        )
    times, values = run_model_uncertainty(record, data_name, cache_path, model_dir, args)
    return times, values, cache_path


def load_bury_ml_probabilities(data_name, bury_prob_root):
    matches = sorted(Path(bury_prob_root).rglob(f"{data_name}_bury_ml_probs.csv"))
    if not matches:
        raise FileNotFoundError(f"Missing Bury ML probabilities for {data_name} under {bury_prob_root}")
    df = pd.read_csv(matches[0])
    missing = [col for col in ["time", *BURY_ML_PROB_COLUMNS] if col not in df.columns]
    if missing:
        raise ValueError(f"{matches[0]} misses columns {missing}")
    return df, matches[0]


def normalize_by_head_mean(values, head_points=40):
    values = np.asarray(values, dtype=float)
    valid_head = values[: min(head_points, len(values))]
    valid_head = valid_head[np.isfinite(valid_head)]
    if len(valid_head) == 0:
        return values
    scale = np.mean(valid_head)
    if abs(scale) < 1e-12:
        return values
    return values / scale


def append_signal_rows(rows, data_name, signal, times, values, component="value"):
    for t, v in zip(np.asarray(times, dtype=float), np.asarray(values, dtype=float)):
        rows.append(
            {
                "data_name": data_name,
                "signal": signal,
                "component": component,
                "time": t,
                "value": v,
            }
        )


def build_signal_data(record, data_name, args):
    ts = as_float_tensor(record["ts_dynamic"]).flatten().numpy()
    ys = as_float_tensor(record["ys_dynamic"])
    if ys.ndim == 1:
        ys = ys.unsqueeze(-1)
    if args.pred_dim >= ys.shape[1]:
        raise ValueError(f"pred_dim={args.pred_dim} invalid for ys_dynamic shape {tuple(ys.shape)}")
    y = ys[:, args.pred_dim].numpy()

    signal_data = {}
    csv_rows = []
    classic_signals = ["ar1", "variance", "sample-entropy-1"]
    if any(sig in args.signals for sig in classic_signals):
        classic = compute_classic_ews(
            ts,
            y,
            method=args.ews_detrend_method,
            span=args.ews_detrend_span,
            bandwidth=args.ews_detrend_bandwidth,
            rolling_window=args.rolling_window,
        )
        for key in classic_signals:
            if key not in args.signals:
                continue
            signal_data[key] = (classic[key].index.to_numpy(dtype=float), classic[key].to_numpy(dtype=float))

    if any(sig in args.signals for sig in ["model_uncertainty", "model_trend"]):
        model_time, model_values, cache_path = load_or_run_model_uncertainty(record, data_name, args)
        model_values = normalize_by_head_mean(model_values, head_points=args.model_normalize_head)
        signal_data["model_uncertainty"] = (model_time, model_values)
        trend_time, trend_values = compute_variance_trend(
            model_time,
            model_values,
            trend_window=args.trend_window,
            min_points=args.trend_min_points,
            normalize_time=True,
        )
        signal_data["model_trend"] = (trend_time, trend_values)
        print(f"{data_name}: model uncertainty from {cache_path}")

    if "bury_ml" in args.signals:
        bury_df, bury_path = load_bury_ml_probabilities(data_name, args.bury_prob_root)
        signal_data["bury_ml"] = bury_df
        print(f"{data_name}: Bury ML probabilities from {bury_path}")

    append_signal_rows(csv_rows, data_name, "trajectory", ts, y, component=f"dim_{args.pred_dim}")
    for signal in args.signals:
        if signal == "bury_ml" and signal in signal_data:
            bury_df = signal_data[signal]
            for col in BURY_ML_PROB_COLUMNS:
                append_signal_rows(csv_rows, data_name, signal, bury_df["time"], bury_df[col], component=col)
        elif signal in signal_data:
            times, values = signal_data[signal]
            append_signal_rows(csv_rows, data_name, signal, times, values)
    return ts, y, signal_data, pd.DataFrame(csv_rows)


def plot_real_data_subfigure(data_name, ts, y, signal_data, transition_time, args):
    n_axes = 1 + len(args.signals)
    fig_height = max(2.2 * n_axes, 5.5)
    fig, axes = plt.subplots(n_axes, 1, figsize=(args.fig_width, fig_height), sharex=True)
    if n_axes == 1:
        axes = [axes]

    axes[0].plot(ts, y, color="#1f4e79", linewidth=1.5)
    axes[0].set_ylabel(args.trajectory_ylabel)
    axes[0].set_title(data_name)

    for axis_index, signal in enumerate(args.signals, start=1):
        ax = axes[axis_index]
        if signal not in signal_data:
            ax.text(0.5, 0.5, f"missing: {signal}", transform=ax.transAxes, ha="center", va="center")
            ax.set_ylabel(signal)
            continue
        if signal == "bury_ml":
            bury_df = signal_data[signal]
            colors = {
                "fold_prob": "#1f77b4",
                "hopf_prob": "#ff7f0e",
                "branch_prob": "#2ca02c",
                "null_prob": "#6c757d",
            }
            labels = {
                "fold_prob": "fold",
                "hopf_prob": "Hopf",
                "branch_prob": "branch",
                "null_prob": "null",
            }
            for col in BURY_ML_PROB_COLUMNS:
                ax.plot(bury_df["time"], bury_df[col], color=colors[col], linewidth=1.1, label=labels[col])
            ax.set_ylim(-0.03, 1.03)
            ax.legend(loc="best", frameon=False, ncol=4, fontsize=8)
            ax.set_ylabel("Bury ML")
        else:
            times, values = signal_data[signal]
            style = {
                "model_uncertainty": ("#d62728", ".", "Model uncertainty"),
                "model_trend": ("#1f77b4", "-.", "Uncertainty trend"),
                "ar1": ("#d62728", "-", "AR(1)"),
                "variance": ("#2ca02c", "-", "Variance"),
                "sample-entropy-1": ("#9467bd", "-", "Sample Entropy"),
            }[signal]
            ax.plot(times, values, linestyle=style[1] if style[1] != "." else "None", marker="." if style[1] == "." else None,
                    color=style[0], linewidth=1.0, markersize=3)
            ax.set_ylabel(style[2])

    if transition_time is not None:
        for ax in axes:
            ax.axvline(transition_time, color="black", linestyle="--", linewidth=1.0, alpha=0.75)

    for ax in axes:
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time")
    axes[-1].set_xlim(float(np.nanmin(ts)), float(np.nanmax(ts)))
    fig.tight_layout()
    return fig


def validate_signals(signals):
    unknown = [sig for sig in signals if sig not in VALID_SIGNALS]
    if unknown:
        raise ValueError(f"Unknown signals {unknown}. Valid signals: {sorted(VALID_SIGNALS)}")


def run_one(data_name_or_path, args):
    data_path = resolve_real_data_path(data_name_or_path, args.data_root)
    record = torch.load(data_path, map_location="cpu")
    data_name = data_name_from_record(record, data_path)
    ts, y, signal_data, signal_table = build_signal_data(record, data_name, args)
    transition_time = get_transition_time_from_record(record)
    output_dir = Path(args.output_root) / data_name
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_real_data_subfigure(data_name, ts, y, signal_data, transition_time, args)
    png_path = output_dir / f"{data_name}_real_data_ews.png"
    svg_path = output_dir / f"{data_name}_real_data_ews.svg"
    fig.savefig(png_path, dpi=args.dpi)
    fig.savefig(svg_path)
    plt.close(fig)
    if not args.no_save_csv:
        signal_table.to_csv(output_dir / f"{data_name}_real_data_ews_signals.csv", index=False)
    print(f"saved: {png_path}")
    print(f"saved: {svg_path}")


def main():
    parser = argparse.ArgumentParser(description="Create real-data EWS subfigures for paper composition.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "dataset" / "real_data")
    parser.add_argument("--real-data-result-root", type=Path, default=PROJECT_ROOT / "ews_results" / "real_data")
    parser.add_argument("--bury-prob-root", type=Path, default=PROJECT_ROOT / "ews_results" / "bury_2021_ml_probs")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "ews_results" / "real_data" / "figures")
    parser.add_argument(
        "--data-real-names",
        nargs="+",
        default=["bury_2021_anoxia_tsid_1"],
        help="Real-data .pt basenames, or explicit .pt paths.",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["model_uncertainty", "model_trend", "ar1", "variance", "bury_ml"],
        help=f"Signals to plot after the trajectory panel. Valid: {sorted(VALID_SIGNALS)}",
    )
    parser.add_argument("--pred-dim", type=int, default=0)
    parser.add_argument("--model-key", choices=sorted(MODEL_KEY_DIRS), default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--model-input", choices=["raw", "detrended"], default="detrended")
    parser.add_argument("--model-detrend-method", choices=["Lowess", "Gaussian", "none"], default="Lowess")
    parser.add_argument("--model-detrend-span", type=float, default=0.2)
    parser.add_argument("--model-detrend-bandwidth", type=float, default=0.2)
    parser.add_argument("--sampling-t", type=float, default=0.1)
    parser.add_argument("--sample-window-step", type=int, default=1)
    parser.add_argument("--parallel-sample", type=int, default=50)
    parser.add_argument("--n-z-samples", type=int, default=100)
    parser.add_argument("--ews-detrend-method", choices=["Lowess", "Gaussian", "none"], default="Gaussian")
    parser.add_argument("--ews-detrend-span", type=float, default=0.2)
    parser.add_argument("--ews-detrend-bandwidth", type=float, default=0.2)
    parser.add_argument("--rolling-window", type=float, default=0.5)
    parser.add_argument("--trend-window", type=int, default=40)
    parser.add_argument("--trend-min-points", type=int, default=5)
    parser.add_argument("--model-normalize-head", type=int, default=40)
    parser.add_argument("--trajectory-ylabel", default="Time Series")
    parser.add_argument("--fig-width", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()
    validate_signals(args.signals)

    for data_name in args.data_real_names:
        run_one(data_name, args)


if __name__ == "__main__":
    main()
