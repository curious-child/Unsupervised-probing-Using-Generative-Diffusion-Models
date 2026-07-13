"""Run Bury et al. PNAS deep-learning EWS classifiers on local real-data .pt files.

This script is intentionally separate from the PyTorch diffusion analysis code.
It requires TensorFlow/Keras only when it is executed. The saved probability
files can then be loaded by PyTorch-only comparison scripts.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from ewstools import TimeSeries


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset" / "real_data"
DEFAULT_MODEL_ROOT = (
    PROJECT_ROOT
    / "evaluation_and_analysis"
    / "bury_2021_ml_models"
    / "best_models_tf215"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "ews_results" / "bury_2021_ml_probs"
PROBABILITY_COLUMNS = ["fold_prob", "hopf_prob", "branch_prob", "null_prob"]


def load_keras_model(model_path: Path):
    """Load a Keras classifier lazily so importing this file does not need TensorFlow."""
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.models import model_from_json
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "TensorFlow/Keras is required to run Bury classifiers. "
            "Use the dedicated conda environment created for this baseline."
        ) from exc
    try:
        return load_model(model_path, compile=False, safe_mode=False)
    except ValueError:
        # Some released .keras files contain valid Keras 2.15 configs and HDF5
        # weights, but tf.keras fails to resolve the internal layer paths. Load
        # the architecture and assign each layer's weights explicitly.
        with zipfile.ZipFile(model_path) as archive:
            config = archive.read("config.json").decode("utf-8")
            weights_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
            weights_file.write(archive.read("model.weights.h5"))
            weights_file.close()

        model = model_from_json(config)
        with h5py.File(weights_file.name, "r") as handle:
            used_groups = set()

            def read_group_arrays(group):
                vars_group = group["cell"]["vars"] if "cell" in group else group["vars"]
                return [vars_group[str(i)][()] for i in range(len(vars_group.keys()))]

            def same_shapes(layer, arrays):
                expected = [tuple(weight.shape) for weight in layer.get_weights()]
                actual = [tuple(array.shape) for array in arrays]
                return expected == actual

            for layer in model.layers:
                if not layer.weights:
                    continue
                layers_group = handle["layers"]
                candidate_name = layer.name if layer.name in layers_group else None
                if candidate_name is not None:
                    weights = read_group_arrays(layers_group[candidate_name])
                    if not same_shapes(layer, weights):
                        candidate_name = None
                if candidate_name is None:
                    for group_name in layers_group.keys():
                        if group_name in used_groups:
                            continue
                        weights = read_group_arrays(layers_group[group_name])
                        if same_shapes(layer, weights):
                            candidate_name = group_name
                            break
                if candidate_name is None:
                    raise ValueError(f"Missing weights for layer {layer.name} in {model_path}")
                weights = read_group_arrays(layers_group[candidate_name])
                layer.set_weights(weights)
                used_groups.add(candidate_name)
        return model


def resolve_real_data_path(name_or_path: str, data_root: Path) -> Path:
    path = Path(name_or_path)
    if path.suffix == ".pt" and path.exists():
        return path
    matches = sorted(data_root.rglob(f"{name_or_path}.pt"))
    if not matches:
        raise FileNotFoundError(f"Cannot find {name_or_path}.pt under {data_root}")
    return matches[0]


def tensor_to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def transition_time(record: dict) -> float | None:
    ts = tensor_to_numpy(record.get("ts_dynamic", []))
    if len(ts) == 0:
        return None
    if "transition_index" in record:
        idx = int(record["transition_index"])
        idx = max(0, min(idx, len(ts) - 1))
        return float(ts[idx])
    if "transition_time" in record:
        return float(record["transition_time"])
    return None


def infer_classifier_length(record: dict, data_name: str) -> int:
    text = " ".join(
        str(record.get(k, "")) for k in ["name", "data_type", "record", "source_file"]
    ).lower()
    text = f"{text} {data_name.lower()}"
    if "thermoacoustic" in text:
        return 1500
    if "anoxia" in text:
        return 500
    if "paleoclimate" in text and ("tsid_3" in text or "younger_dryas" in text):
        return 1500
    if "paleoclimate" in text:
        return 500
    return 1500 if int(record.get("num_time_points", 0)) >= 1500 else 500


def prepare_series(record: dict, pred_dim: int, detrend_method: str, span: float, bandwidth: float):
    values = tensor_to_numpy(record["ys_dynamic"]).astype(float)
    times = tensor_to_numpy(record["ts_dynamic"]).astype(float)
    if values.ndim == 1:
        values = values[:, None]
    if pred_dim >= values.shape[1]:
        raise ValueError(f"pred_dim={pred_dim} is invalid for data shape {values.shape}")

    y = values[:, pred_dim]
    finite = np.isfinite(y) & np.isfinite(times)
    y = y[finite]
    times = times[finite]

    order = np.argsort(times)
    y = y[order]
    times = times[order]
    _, unique_idx = np.unique(times, return_index=True)
    y = y[unique_idx]
    times = times[unique_idx]

    transition = transition_time(record)
    series = pd.Series(y, index=times)
    ts = TimeSeries(series, transition=transition)
    method = detrend_method.lower()
    if method == "lowess":
        ts.detrend(method="Lowess", span=span)
    elif method == "gaussian":
        ts.detrend(method="Gaussian", bandwidth=bandwidth)
    elif method in {"none", "raw"}:
        pass
    else:
        raise ValueError(f"Unsupported detrend method: {detrend_method}")
    return ts, y, times, transition


def load_classifiers(model_root: Path, classifier_len: int, max_models: int | None):
    model_dir = model_root / f"len{classifier_len}"
    model_paths = sorted(model_dir.glob(f"best_model_*_len{classifier_len}.keras"))
    model_paths = [p for p in model_paths if p.stat().st_size > 100000]
    if max_models is not None:
        model_paths = model_paths[:max_models]
    if not model_paths:
        raise FileNotFoundError(f"No complete Keras models found in {model_dir}")
    classifiers = []
    for model_path in model_paths:
        classifiers.append((model_path.stem, load_keras_model(model_path)))
    return classifiers


def apply_bury_classifiers(ts: TimeSeries, classifiers, inc_points: int):
    if len(ts.state) < 2:
        raise ValueError("At least two time points are required.")
    dt = float(ts.state.index[1] - ts.state.index[0])
    inc = dt * inc_points
    for name, classifier in classifiers:
        ts.apply_classifier_inc(classifier, inc=inc, name=name, verbose=0)

    preds = ts.dl_preds.copy()
    if preds.empty:
        raise RuntimeError("Bury classifiers did not produce predictions.")
    grouped = preds.groupby("time")[[0, 1, 2, 3]].mean().reset_index()
    grouped.columns = ["time", *PROBABILITY_COLUMNS]
    grouped["transition_prob"] = grouped[["fold_prob", "hopf_prob", "branch_prob"]].sum(axis=1)
    return grouped


def save_probability_outputs(output_dir: Path, basename: str, table: pd.DataFrame, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{basename}_bury_ml_probs.csv"
    pt_path = output_dir / f"{basename}_bury_ml_probs.pt"
    json_path = output_dir / f"{basename}_bury_ml_probs_metadata.json"
    table.to_csv(csv_path, index=False)
    torch.save(
        {
            "time": torch.tensor(table["time"].to_numpy(dtype=np.float32)),
            "probabilities": torch.tensor(table[PROBABILITY_COLUMNS].to_numpy(dtype=np.float32)),
            "transition_probability": torch.tensor(
                table["transition_prob"].to_numpy(dtype=np.float32)
            ),
            "probability_columns": PROBABILITY_COLUMNS,
            **metadata,
        },
        pt_path,
    )
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, pt_path, json_path


def run_one(args, data_name: str):
    data_path = resolve_real_data_path(data_name, args.data_root)
    record = torch.load(data_path, map_location="cpu")
    basename = Path(record.get("name", data_path.stem)).stem
    classifier_len = args.classifier_len or infer_classifier_length(record, basename)
    ts, raw_values, raw_times, trans_time = prepare_series(
        record,
        pred_dim=args.pred_dim,
        detrend_method=args.detrend_method,
        span=args.detrend_span,
        bandwidth=args.detrend_bandwidth,
    )
    if not args.full_prefix:
        if trans_time is not None:
            pre_transition = ts.state[ts.state.index <= trans_time]
            if len(pre_transition) >= 2:
                ts.state = pre_transition.iloc[-classifier_len:].copy()
        else:
            ts.state = ts.state.iloc[-classifier_len:].copy()
    classifiers = load_classifiers(args.model_root, classifier_len, args.max_models)
    table = apply_bury_classifiers(ts, classifiers, args.inc_points)
    metadata = {
        "name": basename,
        "source_pt": str(data_path),
        "data_type": record.get("data_type", ""),
        "record": record.get("record", ""),
        "pred_dim": args.pred_dim,
        "classifier_len": classifier_len,
        "num_classifiers": len(classifiers),
        "classifier_names": [name for name, _ in classifiers],
        "detrend_method": args.detrend_method,
        "detrend_span": args.detrend_span,
        "detrend_bandwidth": args.detrend_bandwidth,
        "inc_points": args.inc_points,
        "transition_time": trans_time,
        "probability_columns": PROBABILITY_COLUMNS,
    }
    return save_probability_outputs(args.output_root / basename, basename, table, metadata)


def main():
    parser = argparse.ArgumentParser(description="Run Bury et al. ML EWS classifiers.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--data-real-names",
        nargs="+",
        default=[
            "bury_2021_anoxia_tsid_1",
             "bury_2021_anoxia_tsid_2","bury_2021_anoxia_tsid_3","bury_2021_anoxia_tsid_4","bury_2021_anoxia_tsid_5","bury_2021_anoxia_tsid_6",
            "bury_2021_anoxia_tsid_7", "bury_2021_anoxia_tsid_8", "bury_2021_anoxia_tsid_9", "bury_2021_anoxia_tsid_10",
            "bury_2021_anoxia_tsid_11", "bury_2021_anoxia_tsid_12", "bury_2021_anoxia_tsid_13",
            # "bury_2021_paleoclimate_tsid_3_end_of_younger_dryas",
             "bury_2021_thermoacoustic_tsid_1","bury_2021_thermoacoustic_tsid_2","bury_2021_thermoacoustic_tsid_3","bury_2021_thermoacoustic_tsid_4","bury_2021_thermoacoustic_tsid_5",
            "bury_2021_thermoacoustic_tsid_6", "bury_2021_thermoacoustic_tsid_7", "bury_2021_thermoacoustic_tsid_8",
            "bury_2021_thermoacoustic_tsid_9","bury_2021_thermoacoustic_tsid_10","bury_2021_thermoacoustic_tsid_11","bury_2021_thermoacoustic_tsid_12","bury_2021_thermoacoustic_tsid_13",
        ],
    )
    parser.add_argument("--pred-dim", type=int, default=0)
    parser.add_argument("--classifier-len", type=int, choices=[500, 1500], default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--inc-points", type=int, default=10)
    parser.add_argument("--detrend-method", choices=["lowess", "gaussian", "none"], default="lowess")
    parser.add_argument("--detrend-span", type=float, default=0.2)
    parser.add_argument("--detrend-bandwidth", type=float, default=0.2)
    parser.add_argument(
        "--full-prefix",
        action="store_true",
        help="Use the full time series prefix. By default the final classifier-length points are used, matching Bury empirical scripts.",
    )
    args = parser.parse_args()

    for data_name in args.data_real_names:
        csv_path, pt_path, json_path = run_one(args, data_name)
        print(f"saved: {csv_path}")
        print(f"saved: {pt_path}")
        print(f"saved: {json_path}")


if __name__ == "__main__":
    main()
