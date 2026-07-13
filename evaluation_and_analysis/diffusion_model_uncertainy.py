import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for dll_dir in (Path(sys.executable).parent / "bin", Path(sys.executable).parent / "Library" / "bin"):
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))




NETWORK_DYNAMICS = {"SIS", "neuronal", "biomass"}
DEFAULT_SAMPLE_WINDOW_STEP = {
    "SIS": 50,
    "neuronal": 5,
    "biomass": 5,
    "SLBP": 10,
}
DEFAULT_SAMPLING_T = {
    "SIS": 0.1,
    "neuronal": 10,
    "biomass": 10,
    "SLBP": 100,
}




def _as_path(path):
    if path is None:
        return None
    return Path(path)


def _resolve_project_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _dynamic_name(dynamic_type):
    if dynamic_type is None:
        return None
    text = str(dynamic_type)
    if text.lower() == "sis":
        return "SIS"
    if text.lower() == "slbp":
        return "SLBP"
    if text.lower() == "neuronal":
        return "neuronal"
    if text.lower() == "biomass":
        return "biomass"
    return text


def _infer_dynamic_type(data_file=None, loaded_data=None):
    if loaded_data is not None and "N_values" in loaded_data:
        return "SLBP"
    if loaded_data is not None and "tp_values" in loaded_data:
        return None
    if data_file is None:
        return None
    text = str(data_file).replace("\\", "/").lower()
    if "slbp" in text:
        return "SLBP"
    if "sis" in text:
        return "SIS"
    if "neuronal" in text:
        return "neuronal"
    if "biomass" in text:
        return "biomass"
    return None


def load_dynamic_data(data_file, dynamic_type=None, map_location="cpu"):
    loaded_data = torch.load(data_file, map_location=map_location)
    inferred = _infer_dynamic_type(data_file=data_file, loaded_data=loaded_data)
    dynamic_type = _dynamic_name(dynamic_type) or inferred
    if "ys_dynamic" not in loaded_data or "ts_dynamic" not in loaded_data:
        raise KeyError("data_file must contain 'ys_dynamic' and 'ts_dynamic'.")
    torch_time_series = normalize_time_series(
        loaded_data["ys_dynamic"],
        dynamic_type=dynamic_type,
    )
    return {
        "torch_time_series": torch_time_series,
        "time_data": loaded_data["ts_dynamic"],
        "dynamic_type": dynamic_type,
        "loaded_data": loaded_data,
    }


def normalize_time_series(torch_time_series, dynamic_type=None):
    dynamic_type = _dynamic_name(dynamic_type)
    data = torch.as_tensor(torch_time_series).float()
    if data.ndim == 3:
        return data
    if data.ndim != 2:
        raise ValueError("time series must have shape [Node, T, F], [T, F], or [T, Node].")

    if dynamic_type in NETWORK_DYNAMICS:
        return data.t().unsqueeze(-1)
    return data.unsqueeze(0)


def sampling_interval_from_t(sampling_t):
    sampling_t_min = 0.1
    if sampling_t is None:
        return 1
    if sampling_t <= sampling_t_min:
        return 1
    return max(1, int(sampling_t / sampling_t_min))


def sample_time_series(torch_time_series, time_data, sampling_t, return_numpy_time=True):
    interval = sampling_interval_from_t(sampling_t)
    sampled_series = torch_time_series[:, ::interval, :]
    sampled_time = torch.as_tensor(time_data)[::interval]
    if return_numpy_time:
        sampled_time = sampled_time.cpu().detach().numpy()
    return sampled_series, sampled_time


def build_sliding_windows(torch_time_series, time_data, windows, sample_window_step):
    if torch_time_series.ndim != 3:
        raise ValueError("torch_time_series must have shape [Node_num, T_obs_num, F].")
    if torch_time_series.shape[1] < windows:
        raise ValueError(
            "T_obs_num ({}) is shorter than windows ({}).".format(torch_time_series.shape[1], windows)
        )
    window_tensor = torch_time_series.unfold(1, windows, sample_window_step)
    window_tensor = window_tensor.permute(0, 1, 3, 2)
    time_points = np.asarray(time_data)[windows - 1 :: sample_window_step]
    return window_tensor.unbind(1), time_points


def default_sample_window_step(dynamic_type, task_model=None, dataset_config=None):
    dataset_config = dataset_config or {}
    if task_model == "DiffSTG" and dataset_config.get("interval_step") is not None:
        return dataset_config["interval_step"]
    return DEFAULT_SAMPLE_WINDOW_STEP.get(dynamic_type, 10)


def sliding_window_count(sampled_length, windows, sample_window_step):
    if sampled_length < windows:
        return 0
    return (sampled_length - windows) // sample_window_step + 1


def infer_sample_window_step_from_cache(sampled_length, windows, cache_len, fallback_step):
    if cache_len <= 0 or sampled_length < windows:
        return fallback_step
    if sliding_window_count(sampled_length, windows, fallback_step) == cache_len:
        return fallback_step
    if cache_len == 1:
        return fallback_step

    max_offset = sampled_length - windows
    lower_exclusive = max_offset / cache_len
    upper_inclusive = max_offset / (cache_len - 1)
    low = int(np.floor(lower_exclusive)) + 1
    high = int(np.floor(upper_inclusive))
    candidates = [
        step for step in range(max(1, low), max(1, high) + 1)
        if sliding_window_count(sampled_length, windows, step) == cache_len
    ]
    if not candidates:
        return fallback_step
    return min(candidates, key=lambda step: (abs(step - fallback_step), -step))


def read_model_config(model_save_file):
    config_path = Path(model_save_file) / "model_trained.yaml"
    if not config_path.exists():
        raise FileNotFoundError("model config not found: {}".format(config_path))
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_from_dir(model_save_file, device=None, infer_params=None, method_config=None):
    from utils.utils import load_diffusion_model

    model_save_file = Path(model_save_file)
    method_config = method_config or read_model_config(model_save_file)
    train_model_select = None
    if method_config.get("train") is not None:
        train_model_select = method_config["train"].get("train_model_select")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_save_file / "model_trained"
    old_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        model, loaded_net_param = load_diffusion_model(
            str(model_path),
            device=device,
            infer_para=infer_params,
            train_model_select=train_model_select,
        )
    finally:
        os.chdir(old_cwd)
    model.eval()
    return model, loaded_net_param


def default_cache_dir(model_save_file, dynamic_type):
    if model_save_file is not None:
        return Path(model_save_file)
    model_name = "model"
    dynamic_name = _dynamic_name(dynamic_type) or "unknown"
    return PROJECT_ROOT / "ews_results" / "model_uncertainy_cache" / model_name / dynamic_name


def data_cache_name(data_file, suffix=""):
    if data_file is None:
        stem = "data"
        suffix_text = ".pt"
    else:
        data_path = Path(data_file)
        stem = data_path.stem
        suffix_text = data_path.suffix or ".pt"
    return "{}{}{}".format(stem, suffix, suffix_text)


def resolve_cache_path(cache_path, model_save_file, data_file, dynamic_type, suffix=""):
    if cache_path is None:
        cache_dir = default_cache_dir(model_save_file, dynamic_type)
        return cache_dir / data_cache_name(data_file, suffix=suffix)

    cache_path = _resolve_project_path(cache_path)
    if cache_path.suffix == ".pt":
        return cache_path
    return cache_path / data_cache_name(data_file, suffix=suffix)


def resolve_figure_path(cache_file_path):
    return Path(cache_file_path).with_suffix(".png")


def _save_tensor_list(data_list, cache_path):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        torch.save(data_list, f)


def _load_tensor_list(cache_path):
    with open(cache_path, "rb") as f:
        data_list = torch.load(f, map_location="cpu")
    if not isinstance(data_list, list):
        raise TypeError("cache file must contain a list of tensors: {}".format(cache_path))
    return data_list


def _feature_inverse_transform(pred_future, model=None):
    if model is None or getattr(model, "scaler", None) is None:
        return pred_future
    if not hasattr(model, "scaler_mean") or not hasattr(model, "scaler_std"):
        if hasattr(model, "scaler_inverse_transform"):
            return model.scaler_inverse_transform(pred_future)
        return pred_future

    mean = model.scaler_mean.detach().to(pred_future.device, pred_future.dtype)
    std = model.scaler_std.detach().to(pred_future.device, pred_future.dtype)
    if pred_future.ndim >= 3 and pred_future.shape[-2] == mean.numel():
        shape = [1] * pred_future.ndim
        shape[-2] = mean.numel()
        return pred_future * std.view(*shape) + mean.view(*shape)
    if pred_future.shape[-1] == mean.numel():
        return pred_future * std + mean
    return pred_future


def summarize_pred_future_list(pred_future_list, model=None):
    pred_mean_list = []
    uncertainty_ews_list = []
    for pred_future in pred_future_list:
        pred_future = torch.as_tensor(pred_future).detach().cpu()
        pred_future = _feature_inverse_transform(pred_future, model=model)
        if pred_future.ndim == 3:
            pred_future = pred_future.unsqueeze(0)
        if pred_future.ndim != 4:
            raise ValueError(
                "pred_future must have shape [Node, pred_len, F, n_z_samples], got {}".format(
                    tuple(pred_future.shape)
                )
            )
        pred_uncertainty = pred_future.var(dim=-1, unbiased=False)
        uncertainty_ews_list.append(pred_uncertainty.mean().cpu().detach().numpy())
        pred_mean_list.append(pred_future.mean().cpu().detach().numpy())
    return pred_mean_list, uncertainty_ews_list


def summarize_nsdiff_g_list(g_list, pred_dim=0):
    ews_list = []
    pred_mean_list = []
    for gx in g_list:
        gx = torch.as_tensor(gx).detach().cpu()
        if gx.ndim == 2:
            gx = gx.unsqueeze(0)
        if gx.ndim != 3:
            raise ValueError("NsDiff-g cache elements must have shape [Node, pred_len, F].")
        if pred_dim >= gx.shape[-1]:
            raise IndexError("pred_dim {} out of bounds for F={}.".format(pred_dim, gx.shape[-1]))
        node_feature_signal = gx.mean(dim=1)
        ews_list.append(node_feature_signal[:, pred_dim].mean().cpu().detach().numpy())
        pred_mean_list.append(gx.mean().cpu().detach().numpy())
    return pred_mean_list, ews_list


def run_evaluation_cache(model, timeseries_datas, pred_len, cache_path, device, force_recompute=False, max_windows=None):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        return _load_tensor_list(cache_path)

    pred_future_list = []
    iterable = timeseries_datas[:max_windows] if max_windows is not None else timeseries_datas
    with torch.no_grad():
        for time_series in tqdm(iterable, leave=False):
            if model.scaler is not None:
                time_series_trans = model.scaler_transform(time_series.to(device))
            time_series = time_series_trans.clone().to(device)
            pred_future, _ = model.evaluation_step(time_series)
            pred_future = pred_future[:, -pred_len:, :, :].detach().cpu()
            pred_future_list.append(pred_future)
    _save_tensor_list(pred_future_list, cache_path)
    return pred_future_list


def load_diffstg_graph(graph_file):
    import networkx as nx
    import torch_geometric

    if graph_file is None:
        raise ValueError("graph_file is required for DiffSTG.")
    graph_file = _resolve_project_path(graph_file)
    nx_g = nx.read_graphml(graph_file)
    nx_g = nx.convert_node_labels_to_integers(nx_g)
    return torch_geometric.utils.from_networkx(nx_g, group_node_attrs=None)


def normalize_diffstg_pred_future_list(pred_future_list):
    normalized_list = []
    for pred_future in pred_future_list:
        pred_future = torch.as_tensor(pred_future).detach().cpu()
        if pred_future.ndim == 3:
            pred_future = pred_future.unsqueeze(-2)
        if pred_future.ndim != 4:
            raise ValueError(
                "DiffSTG pred_future must have shape [Node, pred_len, F, samples] "
                "or legacy [Node, pred_len, samples], got {}".format(tuple(pred_future.shape))
            )
        normalized_list.append(pred_future)
    return normalized_list


def run_diffstg_evaluation_cache(
    model,
    timeseries_datas,
    pred_len,
    graph_data,
    cache_path,
    device,
    force_recompute=False,
    max_windows=None,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        return normalize_diffstg_pred_future_list(_load_tensor_list(cache_path))

    pred_future_list = []
    iterable = timeseries_datas[:max_windows] if max_windows is not None else timeseries_datas
    with torch.no_grad():
        for time_series in tqdm(iterable, leave=False):
            graph_data_copy = graph_data.clone()
            if getattr(model, "scaler", None) is not None and hasattr(model, "scaler_transform"):
                time_series = model.scaler_transform(time_series.to(device))
            else:
                time_series = time_series.to(device)
            graph_data_copy.x = time_series.clone().to(device)
            pred_future, _ = model.evaluation_step(graph_data_copy)
            pred_future = pred_future[:, -pred_len:, :, :].detach().cpu()
            pred_future_list.append(pred_future)
    _save_tensor_list(pred_future_list, cache_path)
    return pred_future_list


def run_nsdiff_g_cache(model, timeseries_datas, cache_path, device, pred_dim=0, force_recompute=False, max_windows=None):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        return _load_tensor_list(cache_path)
    if not hasattr(model, "cond_pred_model_g") or model.cond_pred_model_g is None:
        return None

    g_list = []
    iterable = timeseries_datas[:max_windows] if max_windows is not None else timeseries_datas
    with torch.no_grad():
        for time_series in tqdm(iterable, leave=False):
            if getattr(model, "scaler", None) is not None and hasattr(model, "scaler_transform"):
                time_series = model.scaler_transform(time_series.to(device))
            else:
                time_series = time_series.to(device)
            gx = model.cond_pred_model_g(time_series).detach().cpu()
            if gx.ndim == 2:
                gx = gx.unsqueeze(0)
            if pred_dim >= gx.shape[-1]:
                raise IndexError("pred_dim {} out of bounds for F={}.".format(pred_dim, gx.shape[-1]))
            g_list.append(gx)
    _save_tensor_list(g_list, cache_path)
    return g_list


def load_sensitivity_model(model_root, model_name, device=None, infer_params=None):
    from utils.utils import load_diffusion_model

    model_root = _resolve_project_path(model_root)
    config_path = model_root / "models" / "{}.yaml".format(model_name)
    model_path = model_root / "models" / model_name
    if not config_path.exists():
        raise FileNotFoundError("model config not found: {}".format(config_path))
    if not model_path.exists():
        raise FileNotFoundError("model checkpoint not found: {}".format(model_path))

    with open(config_path, "r", encoding="utf-8") as f:
        method_config = yaml.safe_load(f)
    train_model_select = None
    if method_config.get("train") is not None:
        train_model_select = method_config["train"].get("train_model_select")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        model, loaded_net_param = load_diffusion_model(
            str(model_path),
            device=device,
            infer_para=infer_params,
            train_model_select=train_model_select,
        )
    finally:
        os.chdir(old_cwd)
    model.eval()
    return model, method_config, loaded_net_param


def slbp_sensitivity_cache_path(model_root, model_name, data_trend, sample_window_step=10):
    model_root = _resolve_project_path(model_root)
    return model_root / "datas" / "{}_pred_future_{}_{}.pt".format(
        model_name,
        data_trend,
        sample_window_step,
    )


def build_slbp_sensitivity_windows(torch_time_series, time_data, windows, pred_len, sampling_t, sample_window_step):
    sampling_torch_time_series = torch_data_preprocessing_like_slbp(torch_time_series, sampling_t=sampling_t)
    sampled_time = torch_data_preprocessing_like_slbp(time_data, sampling_t=sampling_t, return_numpy=True)

    time_points = sampled_time[windows - 1 :: sample_window_step]
    input_data = sampling_torch_time_series.unfold(0, windows, sample_window_step)
    input_data = input_data.permute(0, 2, 1)
    input_datas = input_data.unbind(0)

    pred_data = sampling_torch_time_series[windows:, :]
    if pred_data.shape[0] >= pred_len:
        pred_data = pred_data.unfold(0, pred_len, sample_window_step)
        pred_data = pred_data.permute(0, 2, 1)
        pred_datas = pred_data.unbind(0)
    else:
        pred_datas = ()
    return input_datas, pred_datas, time_points


def torch_data_preprocessing_like_slbp(time_data, sampling_t, return_numpy=False):
    sampling_interval = sampling_interval_from_t(sampling_t)
    sampled = torch.as_tensor(time_data)[::sampling_interval]
    if return_numpy:
        return sampled.cpu().detach().numpy()
    return sampled


def read_sensitivity_pred_future_cache(cache_path):
    try:
        return _load_tensor_list(cache_path)
    except Exception as exc:
        print("warning: failed to read cache {}, recomputing ({})".format(cache_path, exc))
        return None


def run_slbp_sensitivity_cache(
    model,
    input_datas,
    cache_path,
    device,
    force_recompute=False,
    max_windows=None,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        pred_future_list = read_sensitivity_pred_future_cache(cache_path)
        if pred_future_list is not None:
            return pred_future_list

    pred_future_list = []
    iterable = input_datas[:max_windows] if max_windows is not None else input_datas
    with torch.no_grad():
        for time_series in tqdm(iterable, leave=False):
            if getattr(model, "scaler", None) is not None:
                time_series = model.scaler_transform(time_series.to(device))
            time_series = time_series.clone().unsqueeze(0).to(device)
            pred_future, _ = model.evaluation_step(time_series)
            pred_future_list.append(pred_future.squeeze(0).detach().cpu())
    _save_tensor_list(pred_future_list, cache_path)
    return pred_future_list


def summarize_slbp_sensitivity(pred_future_list, pred_datas, model=None, device=None, pred_dim=0):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpv_list = []
    pred_error_list = []
    for pred_future in pred_future_list:
        pred_future = torch.as_tensor(pred_future).detach().cpu()
        if pred_future.ndim != 3:
            raise ValueError("SLBP sensitivity cache elements must have shape [pred_len, F, n_z_samples].")
        if pred_dim >= pred_future.shape[1]:
            raise IndexError("pred_dim {} out of bounds for F={}.".format(pred_dim, pred_future.shape[1]))
        pred_uncertainty = pred_future.var(dim=-1, unbiased=False).mean(dim=0)
        mpv_list.append(pred_uncertainty[pred_dim].cpu().detach().numpy())

    for pred_future, pred_data in zip(pred_future_list, pred_datas):
        pred_future = torch.as_tensor(pred_future).detach().cpu()
        pred_data = torch.as_tensor(pred_data).detach().cpu()
        if getattr(model, "scaler", None) is not None and hasattr(model, "scaler_transform"):
            pred_data = model.scaler_transform(pred_data.to(device)).to("cpu")
        pred_error = torch.abs(pred_future.mean(dim=-1) - pred_data)
        pred_error_mean = pred_error.mean(dim=0)
        pred_error_list.append(pred_error_mean[pred_dim].cpu().detach().numpy())
    return mpv_list, pred_error_list


def slbp_sensitivity_ews(
    model_root,
    model_name,
    torch_time_series,
    time_data,
    data_trend="increase",
    pred_dim=0,
    sample_window_step=10,
    infer_params=None,
    force_recompute=False,
    max_windows=None,
    device=None,
):
    model_root = _resolve_project_path(model_root)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, method_config, loaded_net_param = load_sensitivity_model(
        model_root,
        model_name,
        device=device,
        infer_params=infer_params,
    )
    dataset_config = method_config.get("dataset", {})
    windows = dataset_config["windows"]
    pred_len = dataset_config["pred_len"]
    sampling_t = dataset_config["sampling_t"]

    input_datas, pred_datas, time_points = build_slbp_sensitivity_windows(
        torch_time_series=torch_time_series,
        time_data=time_data,
        windows=windows,
        pred_len=pred_len,
        sampling_t=sampling_t,
        sample_window_step=sample_window_step,
    )
    cache_path = slbp_sensitivity_cache_path(
        model_root=model_root,
        model_name=model_name,
        data_trend=data_trend,
        sample_window_step=sample_window_step,
    )
    pred_future_list = run_slbp_sensitivity_cache(
        model=model,
        input_datas=input_datas,
        cache_path=cache_path,
        device=device,
        force_recompute=force_recompute,
        max_windows=max_windows,
    )
    mpv_list, prediction_error_list = summarize_slbp_sensitivity(
        pred_future_list=pred_future_list,
        pred_datas=pred_datas,
        model=model,
        device=device,
        pred_dim=pred_dim,
    )
    return {
        "time_points": time_points,
        "mpv": mpv_list,
        "prediction_error": prediction_error_list,
        "pred_future_list": pred_future_list,
        "cache_path": str(cache_path),
        "windows": windows,
        "pred_len": pred_len,
        "sampling_t": sampling_t,
        "sample_window_step": sample_window_step,
        "model_root": str(model_root),
        "model_name": model_name,
        "loaded_net_param": loaded_net_param,
    }


def slbp_fig6_cache_path(model_root, model_name, data_trend, sample_window_step=10, cache_subdir=None, kind="pred_future"):
    model_root = _resolve_project_path(model_root)
    cache_dir = model_root / "datas"
    if cache_subdir:
        cache_dir = cache_dir / cache_subdir
    return cache_dir / "{}_{}_{}_{}.pt".format(
        model_name,
        kind,
        data_trend,
        sample_window_step,
    )


def slbp_fig6_pred_future_gx_cache_path(model_root, model_name, data_trend, sample_window_step=10, cache_subdir=None):
    model_root = _resolve_project_path(model_root)
    cache_dir = model_root / "datas"
    if cache_subdir:
        cache_dir = cache_dir / cache_subdir
    return cache_dir / "{}_pred_future_{}_{}_gx.pt".format(
        model_name,
        data_trend,
        sample_window_step,
    )


def _legacy_single_underscore_model_name(model_name):
    return str(model_name).replace("dataset__", "dataset_", 1)


def _read_slbp_fig6_model_config(model_root, model_name):
    model_root = _resolve_project_path(model_root)
    config_path = model_root / "models" / "{}.yaml".format(model_name)
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _slbp_fig6_dataset_config(model_root, model_name, windows=None, pred_len=None, sampling_t=None):
    method_config = _read_slbp_fig6_model_config(model_root, model_name)
    dataset_config = method_config.get("dataset", {}) if method_config else {}
    return {
        "windows": windows if windows is not None else dataset_config.get("windows", 200),
        "pred_len": pred_len if pred_len is not None else dataset_config.get("pred_len", 200),
        "sampling_t": sampling_t if sampling_t is not None else dataset_config.get("sampling_t", 100),
        "method_config": method_config,
    }


def _slbp_cache_elements_have_ndim(data_list, ndim):
    if not data_list:
        return False
    return all(torch.as_tensor(item).ndim == ndim for item in data_list)


def _slbp_cache_elements_are_gx(data_list):
    return _slbp_cache_elements_have_ndim(data_list, 2) or (
        bool(data_list)
        and all(torch.as_tensor(item).ndim == 3 and torch.as_tensor(item).shape[0] == 1 for item in data_list)
    )


def _slbp_intrinsic_dimension(trajectories):
    trajectories = torch.as_tensor(trajectories, dtype=torch.float32)
    if trajectories.ndim != 2 or trajectories.shape[0] < 2:
        return np.nan
    centered = trajectories - trajectories.mean(dim=0, keepdim=True)
    covariance = centered.T.mm(centered) / max(trajectories.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    eigenvalues = torch.sort(eigenvalues, descending=True).values.clamp_min(0)
    total = eigenvalues.sum()
    if float(total) <= 0:
        return np.nan
    cumulative = torch.cumsum(eigenvalues / total, dim=0)
    return int(torch.where(cumulative >= 0.8)[0][0].item() + 1)


def summarize_slbp_sampling_for_fig6(pred_future_list, pred_dim=0):
    mpv_list = []
    dim_list = []
    for pred_future in pred_future_list:
        pred_future = torch.as_tensor(pred_future).detach().cpu()
        if pred_future.ndim != 3:
            raise ValueError("SLBP sampling cache elements must have shape [pred_len, F, n_z_samples].")
        if pred_dim >= pred_future.shape[1]:
            raise IndexError("pred_dim {} out of bounds for F={}.".format(pred_dim, pred_future.shape[1]))
        pred_uncertainty = pred_future.var(dim=-1, unbiased=False).mean(dim=0)
        mpv_list.append(float(pred_uncertainty[pred_dim].cpu().detach().numpy()))
        traj = pred_future.permute(2, 0, 1).reshape(pred_future.shape[-1], -1)
        dim_list.append(_slbp_intrinsic_dimension(traj))
    return mpv_list, dim_list


def summarize_slbp_gx_for_fig6(gx_list, pred_dim=0):
    gx_mpv = []
    for gx in gx_list:
        gx = torch.as_tensor(gx).detach().cpu()
        if gx.ndim == 3 and gx.shape[0] == 1:
            gx = gx.squeeze(0)
        if gx.ndim != 2:
            raise ValueError("SLBP gx cache elements must have shape [pred_len, F] or [1, pred_len, F].")
        if pred_dim >= gx.shape[-1]:
            raise IndexError("pred_dim {} out of bounds for F={}.".format(pred_dim, gx.shape[-1]))
        gx_mpv.append(float(gx[:, pred_dim].mean().cpu().detach().numpy()))
    return gx_mpv


def run_slbp_gx_cache_for_fig6(
    model,
    input_datas,
    cache_path,
    device,
    pred_dim=0,
    force_recompute=False,
    max_windows=None,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        gx_list = _load_tensor_list(cache_path)
        if _slbp_cache_elements_have_ndim(gx_list, 2) or (
            gx_list and all(torch.as_tensor(item).ndim == 3 and torch.as_tensor(item).shape[0] == 1 for item in gx_list)
        ):
            return gx_list
    if not hasattr(model, "cond_pred_model_g") or model.cond_pred_model_g is None:
        raise ValueError("model does not provide cond_pred_model_g for gx generation.")

    gx_list = []
    iterable = input_datas[:max_windows] if max_windows is not None else input_datas
    with torch.no_grad():
        for time_series in tqdm(iterable, leave=False):
            if getattr(model, "scaler", None) is not None and hasattr(model, "scaler_transform"):
                time_series = model.scaler_transform(time_series.to(device))
            else:
                time_series = time_series.to(device)
            gx = model.cond_pred_model_g(time_series.clone().unsqueeze(0).to(device)).detach().cpu()
            if gx.ndim == 3 and gx.shape[0] == 1:
                gx = gx.squeeze(0)
            if pred_dim >= gx.shape[-1]:
                raise IndexError("pred_dim {} out of bounds for F={}.".format(pred_dim, gx.shape[-1]))
            gx_list.append(gx)
    _save_tensor_list(gx_list, cache_path)
    return gx_list


def slbp_sampling_analysis(
    model_root,
    model_name,
    torch_time_series,
    time_data,
    data_trend="increase",
    pred_dim=0,
    sample_window_step=10,
    cache_subdir=None,
    windows=None,
    pred_len=None,
    sampling_t=None,
    infer_params=None,
    force_recompute=False,
    max_windows=None,
    device=None,
    allow_unavailable=True,
):
    cfg = _slbp_fig6_dataset_config(model_root, model_name, windows=windows, pred_len=pred_len, sampling_t=sampling_t)
    input_datas, _pred_datas, time_points = build_slbp_sensitivity_windows(
        torch_time_series=torch_time_series,
        time_data=time_data,
        windows=cfg["windows"],
        pred_len=cfg["pred_len"],
        sampling_t=cfg["sampling_t"],
        sample_window_step=sample_window_step,
    )
    cache_path = slbp_fig6_cache_path(
        model_root=model_root,
        model_name=model_name,
        data_trend=data_trend,
        sample_window_step=sample_window_step,
        cache_subdir=cache_subdir,
        kind="pred_future",
    )
    sampling_cache_path = slbp_fig6_cache_path(
        model_root=model_root,
        model_name=model_name,
        data_trend=data_trend,
        sample_window_step=sample_window_step,
        cache_subdir=cache_subdir,
        kind="sampling_pred_future",
    )

    try:
        active_cache_path = cache_path
        pred_future_list = None
        if not force_recompute:
            if cache_path.exists():
                candidate_list = _load_tensor_list(cache_path)
                if _slbp_cache_elements_have_ndim(candidate_list, 3):
                    pred_future_list = candidate_list
                else:
                    active_cache_path = sampling_cache_path
            if pred_future_list is None and sampling_cache_path.exists():
                candidate_list = _load_tensor_list(sampling_cache_path)
                if _slbp_cache_elements_have_ndim(candidate_list, 3):
                    pred_future_list = candidate_list
                    active_cache_path = sampling_cache_path
                else:
                    raise ValueError(
                        "sampling cache exists but is not [pred_len, F, n_z_samples]: {}".format(
                            sampling_cache_path
                        )
                    )

        if pred_future_list is None:
            if cache_path.exists() and active_cache_path == cache_path:
                active_cache_path = sampling_cache_path
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, _method_config, _loaded_net_param = load_sensitivity_model(
                model_root,
                model_name,
                device=device,
                infer_params=infer_params,
            )
            pred_future_list = run_slbp_sensitivity_cache(
                model=model,
                input_datas=input_datas,
                cache_path=active_cache_path,
                device=device,
                force_recompute=force_recompute,
                max_windows=max_windows,
            )
            if not _slbp_cache_elements_have_ndim(pred_future_list, 3):
                raise ValueError(
                    "generated sampling cache is not [pred_len, F, n_z_samples]: {}".format(active_cache_path)
                )
        mpv, intrinsic_dimension = summarize_slbp_sampling_for_fig6(pred_future_list, pred_dim=pred_dim)
        return {
            "available": True,
            "time_points": time_points[: len(mpv)],
            "mpv": mpv,
            "intrinsic_dimension": intrinsic_dimension,
            "pred_future_list": pred_future_list,
            "cache_path": str(active_cache_path),
            "windows": cfg["windows"],
            "pred_len": cfg["pred_len"],
            "sampling_t": cfg["sampling_t"],
            "sample_window_step": sample_window_step,
            "reason": "",
        }
    except Exception as exc:
        if not allow_unavailable:
            raise
        return {
            "available": False,
            "time_points": time_points,
            "mpv": [],
            "intrinsic_dimension": [],
            "pred_future_list": None,
            "cache_path": str(sampling_cache_path if cache_path.exists() else cache_path),
            "windows": cfg["windows"],
            "pred_len": cfg["pred_len"],
            "sampling_t": cfg["sampling_t"],
            "sample_window_step": sample_window_step,
            "reason": str(exc),
        }


def slbp_gx_analysis(
    model_root,
    model_name,
    torch_time_series,
    time_data,
    data_trend="increase",
    pred_dim=0,
    sample_window_step=10,
    cache_subdir=None,
    windows=None,
    pred_len=None,
    sampling_t=None,
    infer_params=None,
    force_recompute=False,
    max_windows=None,
    device=None,
):
    cfg = _slbp_fig6_dataset_config(model_root, model_name, windows=windows, pred_len=pred_len, sampling_t=sampling_t)
    input_datas, _pred_datas, time_points = build_slbp_sensitivity_windows(
        torch_time_series=torch_time_series,
        time_data=time_data,
        windows=cfg["windows"],
        pred_len=cfg["pred_len"],
        sampling_t=cfg["sampling_t"],
        sample_window_step=sample_window_step,
    )
    gx_cache_path = slbp_fig6_pred_future_gx_cache_path(
        model_root=model_root,
        model_name=model_name,
        data_trend=data_trend,
        sample_window_step=sample_window_step,
        cache_subdir=cache_subdir,
    )
    old_gx_cache_path = slbp_fig6_cache_path(
        model_root=model_root,
        model_name=model_name,
        data_trend=data_trend,
        sample_window_step=sample_window_step,
        cache_subdir=cache_subdir,
        kind="gx",
    )
    legacy_cache_path = slbp_fig6_cache_path(
        model_root=model_root,
        model_name=model_name,
        data_trend=data_trend,
        sample_window_step=sample_window_step,
        cache_subdir=cache_subdir,
        kind="pred_future",
    )
    legacy_name_cache_path = slbp_fig6_cache_path(
        model_root=model_root,
        model_name=_legacy_single_underscore_model_name(model_name),
        data_trend=data_trend,
        sample_window_step=sample_window_step,
        cache_subdir=cache_subdir,
        kind="pred_future",
    )

    if gx_cache_path.exists() and not force_recompute:
        gx_list = _load_tensor_list(gx_cache_path)
    elif old_gx_cache_path.exists() and not force_recompute:
        gx_list = _load_tensor_list(old_gx_cache_path)
        if not _slbp_cache_elements_are_gx(gx_list):
            raise ValueError("gx cache exists but is not a gx cache: {}".format(old_gx_cache_path))
        gx_cache_path = old_gx_cache_path
    elif legacy_cache_path.exists() and not force_recompute:
        gx_list = _load_tensor_list(legacy_cache_path)
        if _slbp_cache_elements_are_gx(gx_list):
            gx_cache_path = legacy_cache_path
        else:
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, _method_config, _loaded_net_param = load_sensitivity_model(
                model_root,
                model_name,
                device=device,
                infer_params=infer_params,
            )
            gx_list = run_slbp_gx_cache_for_fig6(
                model=model,
                input_datas=input_datas,
                cache_path=gx_cache_path,
                device=device,
                pred_dim=pred_dim,
                force_recompute=force_recompute,
                max_windows=max_windows,
            )
    elif legacy_name_cache_path.exists() and not force_recompute:
        gx_list = _load_tensor_list(legacy_name_cache_path)
        if _slbp_cache_elements_are_gx(gx_list):
            gx_cache_path = legacy_name_cache_path
        else:
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, _method_config, _loaded_net_param = load_sensitivity_model(
                model_root,
                model_name,
                device=device,
                infer_params=infer_params,
            )
            gx_list = run_slbp_gx_cache_for_fig6(
                model=model,
                input_datas=input_datas,
                cache_path=gx_cache_path,
                device=device,
                pred_dim=pred_dim,
                force_recompute=force_recompute,
                max_windows=max_windows,
            )
    else:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _method_config, _loaded_net_param = load_sensitivity_model(
            model_root,
            model_name,
            device=device,
            infer_params=infer_params,
        )
        gx_list = run_slbp_gx_cache_for_fig6(
            model=model,
            input_datas=input_datas,
            cache_path=gx_cache_path,
            device=device,
            pred_dim=pred_dim,
            force_recompute=force_recompute,
            max_windows=max_windows,
        )

    gx_mpv = summarize_slbp_gx_for_fig6(gx_list, pred_dim=pred_dim)
    return {
        "time_points": time_points[: len(gx_mpv)],
        "gx_mpv": gx_mpv,
        "gx_list": gx_list,
        "cache_path": str(gx_cache_path),
        "windows": cfg["windows"],
        "pred_len": cfg["pred_len"],
        "sampling_t": cfg["sampling_t"],
        "sample_window_step": sample_window_step,
    }


def summarize_slbp_mpv_cache_for_fig5(data_list, pred_dim=0):
    if _slbp_cache_elements_have_ndim(data_list, 3):
        mpv, _dimension = summarize_slbp_sampling_for_fig6(data_list, pred_dim=pred_dim)
        return mpv, "sampling"
    if _slbp_cache_elements_are_gx(data_list):
        return summarize_slbp_gx_for_fig6(data_list, pred_dim=pred_dim), "gx"
    if data_list:
        shape = tuple(torch.as_tensor(data_list[0]).shape)
    else:
        shape = None
    raise ValueError("Unsupported SLBP MPV cache element shape: {}".format(shape))


def slbp_mpv_analysis(
    model_root,
    model_name,
    torch_time_series,
    time_data,
    cache_path,
    pred_dim=0,
    sample_window_step=10,
    windows=None,
    pred_len=None,
    sampling_t=None,
    infer_params=None,
    force_recompute=False,
    max_windows=None,
    device=None,
):
    cfg = _slbp_fig6_dataset_config(model_root, model_name, windows=windows, pred_len=pred_len, sampling_t=sampling_t)
    cache_path = _resolve_project_path(cache_path)
    sampled_time = torch_data_preprocessing_like_slbp(time_data, sampling_t=cfg["sampling_t"], return_numpy=True)

    if cache_path.exists() and not force_recompute:
        data_list = _load_tensor_list(cache_path)
        inferred_step = infer_sample_window_step_from_cache(
            sampled_length=len(sampled_time),
            windows=cfg["windows"],
            cache_len=len(data_list),
            fallback_step=sample_window_step,
        )
        mpv, source = summarize_slbp_mpv_cache_for_fig5(data_list, pred_dim=pred_dim)
        return {
            "time_points": sampled_time[cfg["windows"] - 1 :: inferred_step][: len(mpv)],
            "mpv": mpv,
            "pred_future_list": data_list,
            "cache_path": str(cache_path),
            "windows": cfg["windows"],
            "pred_len": cfg["pred_len"],
            "sampling_t": cfg["sampling_t"],
            "sample_window_step": inferred_step,
            "uncertainty_source": source,
        }

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _method_config, _loaded_net_param = load_sensitivity_model(
        model_root,
        model_name,
        device=device,
        infer_params=infer_params,
    )
    input_datas, _pred_datas, time_points = build_slbp_sensitivity_windows(
        torch_time_series=torch_time_series,
        time_data=time_data,
        windows=cfg["windows"],
        pred_len=cfg["pred_len"],
        sampling_t=cfg["sampling_t"],
        sample_window_step=sample_window_step,
    )
    pred_future_list = run_slbp_sensitivity_cache(
        model=model,
        input_datas=input_datas,
        cache_path=cache_path,
        device=device,
        force_recompute=force_recompute,
        max_windows=max_windows,
    )
    mpv, source = summarize_slbp_mpv_cache_for_fig5(pred_future_list, pred_dim=pred_dim)
    return {
        "time_points": time_points[: len(mpv)],
        "mpv": mpv,
        "pred_future_list": pred_future_list,
        "cache_path": str(cache_path),
        "windows": cfg["windows"],
        "pred_len": cfg["pred_len"],
        "sampling_t": cfg["sampling_t"],
        "sample_window_step": sample_window_step,
        "uncertainty_source": source,
    }


def slbp_direct_model_cache_analysis(
    model_save_file,
    torch_time_series,
    time_data,
    cache_path,
    pred_dim=0,
    sample_window_step=10,
    cache_kind="auto",
    infer_params=None,
    force_recompute=False,
    max_windows=None,
    device=None,
    compute_prediction_error=False,
):
    method_config = read_model_config(model_save_file)
    dataset_cfg = method_config.get("dataset", {})
    windows = int(dataset_cfg.get("windows", method_config.get("net", {}).get("windows", 200)))
    pred_len = int(dataset_cfg.get("pred_len", method_config.get("net", {}).get("pred_len", 200)))
    sampling_t = int(dataset_cfg.get("sampling_t", 100))
    cache_path = _resolve_project_path(cache_path)
    sampled_time = torch_data_preprocessing_like_slbp(time_data, sampling_t=sampling_t, return_numpy=True)

    data_list = None
    source = None
    model = None
    if cache_path.exists() and not force_recompute:
        data_list = _load_tensor_list(cache_path)
        mpv, source = summarize_slbp_mpv_cache_for_fig5(data_list, pred_dim=pred_dim)
    else:
        if cache_kind not in {"gx", "sampling"}:
            raise ValueError("cache_kind must be 'gx' or 'sampling' when cache is missing.")
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _loaded_net_param = load_model_from_dir(
            model_save_file,
            device=device,
            infer_params=infer_params,
            method_config=method_config,
        )
        input_datas, _pred_datas, _time_points = build_slbp_sensitivity_windows(
            torch_time_series=torch_time_series,
            time_data=time_data,
            windows=windows,
            pred_len=pred_len,
            sampling_t=sampling_t,
            sample_window_step=sample_window_step,
        )
        if cache_kind == "gx":
            data_list = run_slbp_gx_cache_for_fig6(
                model=model,
                input_datas=input_datas,
                cache_path=cache_path,
                device=device,
                pred_dim=pred_dim,
                force_recompute=force_recompute,
                max_windows=max_windows,
            )
        else:
            data_list = run_slbp_sensitivity_cache(
                model=model,
                input_datas=input_datas,
                cache_path=cache_path,
                device=device,
                force_recompute=force_recompute,
                max_windows=max_windows,
            )
        mpv, source = summarize_slbp_mpv_cache_for_fig5(data_list, pred_dim=pred_dim)

    inferred_step = infer_sample_window_step_from_cache(
        sampled_length=len(sampled_time),
        windows=windows,
        cache_len=len(data_list),
        fallback_step=sample_window_step,
    )
    time_points = sampled_time[windows - 1 :: inferred_step][: len(mpv)]
    result = {
        "time_points": time_points,
        "mpv": mpv,
        "pred_future_list": data_list,
        "cache_path": str(cache_path),
        "windows": windows,
        "pred_len": pred_len,
        "sampling_t": sampling_t,
        "sample_window_step": inferred_step,
        "uncertainty_source": source,
    }

    if compute_prediction_error:
        if source != "sampling":
            raise ValueError("prediction_error requires a sampling cache, got '{}'.".format(source))
        if model is None:
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, _loaded_net_param = load_model_from_dir(
                model_save_file,
                device=device,
                infer_params=infer_params,
                method_config=method_config,
            )
        _input_datas, pred_datas, _time_points = build_slbp_sensitivity_windows(
            torch_time_series=torch_time_series,
            time_data=time_data,
            windows=windows,
            pred_len=pred_len,
            sampling_t=sampling_t,
            sample_window_step=inferred_step,
        )
        _mpv, prediction_error = summarize_slbp_sensitivity(
            pred_future_list=data_list,
            pred_datas=pred_datas[: len(data_list)],
            model=model,
            device=device,
            pred_dim=pred_dim,
        )
        result["prediction_error"] = prediction_error
    return result


def slbp_raw_window_variance(torch_time_series, time_data, windows=200, sampling_t=100, sample_window_step=10, pred_dim=0):
    sampled_series = torch_data_preprocessing_like_slbp(torch_time_series, sampling_t=sampling_t)
    sampled_time = torch_data_preprocessing_like_slbp(time_data, sampling_t=sampling_t, return_numpy=True)
    if sampled_series.ndim != 2:
        raise ValueError("SLBP raw series must have shape [T, F].")
    if pred_dim >= sampled_series.shape[1]:
        raise IndexError("pred_dim {} out of bounds for F={}.".format(pred_dim, sampled_series.shape[1]))
    series = sampled_series[:, pred_dim]
    windows_tensor = series.unfold(0, windows, sample_window_step)
    variances = windows_tensor.var(dim=1, unbiased=False).cpu().detach().numpy()
    time_points = sampled_time[windows - 1 :: sample_window_step][: len(variances)]
    return {
        "time_points": time_points,
        "variance": variances,
        "windows": windows,
        "sampling_t": sampling_t,
        "sample_window_step": sample_window_step,
    }


def uncertainty_ews(
    model_save_file=None,
    data_file=None,
    torch_time_series=None,
    time_data=None,
    dynamic_type=None,
    task_model=None,
    graph_file=None,
    cache_path=None,
    sample_window_step=None,
    sampling_t=None,
    infer_params=None,
    pred_dim=0,
    force_recompute=False,
    save_nsdiff_g=True,
    nsdiff_g_path=None,
    uncertainty_method="sampling",
    max_windows=None,
    device=None,
    load_model_when_cached=False,
):
    dynamic_type = _dynamic_name(dynamic_type)
    uncertainty_method = str(uncertainty_method).lower()
    method_aliases = {
        "variance": "sampling",
        "sampling_variance": "sampling",
        "pred_future": "sampling",
        "pred": "sampling",
        "g": "gx",
        "preg": "gx",
        "nsdiff_g": "gx",
    }
    uncertainty_method = method_aliases.get(uncertainty_method, uncertainty_method)
    if uncertainty_method not in {"sampling", "gx", "both"}:
        raise ValueError("uncertainty_method must be one of: sampling, gx, both.")

    loaded_data = None
    if data_file is not None:
        data_file = _resolve_project_path(data_file)
        loaded_data = load_dynamic_data(data_file, dynamic_type=dynamic_type)
        torch_time_series = loaded_data["torch_time_series"]
        time_data = loaded_data["time_data"]
        dynamic_type = _dynamic_name(dynamic_type) or loaded_data["dynamic_type"]
    elif torch_time_series is not None:
        dynamic_type = _dynamic_name(dynamic_type)
        torch_time_series = normalize_time_series(torch_time_series, dynamic_type=dynamic_type)
    else:
        raise ValueError("Provide data_file or torch_time_series.")

    if time_data is None:
        raise ValueError("time_data is required when data_file is not provided.")

    method_config = None
    model = None
    loaded_net_param = None
    if model_save_file is not None:
        model_save_file = _resolve_project_path(model_save_file)
        method_config = read_model_config(model_save_file)

    if task_model is None and method_config is not None:
        task_model = method_config.get("net", {}).get("task_model")

    dataset_config = method_config.get("dataset", {}) if method_config else {}
    windows = dataset_config.get("windows")
    pred_len = dataset_config.get("pred_len")
    if windows is None or pred_len is None:
        raise ValueError("model_trained.yaml must provide dataset.windows and dataset.pred_len.")

    cache_path = resolve_cache_path(
        cache_path=cache_path,
        model_save_file=model_save_file,
        data_file=data_file,
        dynamic_type=dynamic_type,
    )
    need_sampling_uncertainty = uncertainty_method in {"sampling", "both"}
    need_gx_uncertainty = uncertainty_method in {"gx", "both"} or (
        save_nsdiff_g and uncertainty_method == "sampling"
    )
    nsdiff_path = None
    if need_gx_uncertainty:
        if nsdiff_g_path is not None:
            nsdiff_path = resolve_cache_path(
                cache_path=nsdiff_g_path,
                model_save_file=model_save_file,
                data_file=data_file,
                dynamic_type=dynamic_type,
                suffix="_gx",
            )
        else:
            nsdiff_path = resolve_cache_path(
                cache_path=cache_path.parent,
                model_save_file=model_save_file,
                data_file=data_file,
                dynamic_type=dynamic_type,
                suffix="_gx",
            )

    cached_pred_future_list = None
    if need_sampling_uncertainty and cache_path.exists() and not force_recompute:
        if task_model == "DiffSTG":
            cached_pred_future_list = normalize_diffstg_pred_future_list(_load_tensor_list(cache_path))
        else:
            cached_pred_future_list = _load_tensor_list(cache_path)
    cached_nsdiff_g_list = None
    if need_gx_uncertainty and nsdiff_path is not None and nsdiff_path.exists() and not force_recompute:
        cached_nsdiff_g_list = _load_tensor_list(nsdiff_path)

    if sampling_t is None:
        sampling_t = dataset_config.get("sampling_t", DEFAULT_SAMPLING_T.get(dynamic_type, 0.1))
    sampled_series, sampled_time = sample_time_series(torch_time_series, time_data, sampling_t=sampling_t)

    if sample_window_step is None:
        fallback_step = default_sample_window_step(
            dynamic_type=dynamic_type,
            task_model=task_model,
            dataset_config=dataset_config,
        )
        step_cache_len = None
        if cached_pred_future_list is not None:
            step_cache_len = len(cached_pred_future_list)
        elif cached_nsdiff_g_list is not None:
            step_cache_len = len(cached_nsdiff_g_list)
        if step_cache_len is not None:
            sample_window_step = infer_sample_window_step_from_cache(
                sampled_length=sampled_series.shape[1],
                windows=windows,
                cache_len=step_cache_len,
                fallback_step=fallback_step,
            )
        else:
            sample_window_step = fallback_step

    timeseries_datas, time_points = build_sliding_windows(
        sampled_series,
        sampled_time,
        windows=windows,
        sample_window_step=sample_window_step,
    )

    if need_sampling_uncertainty:
        if task_model == "DiffSTG":
            if dynamic_type not in NETWORK_DYNAMICS:
                raise ValueError("DiffSTG only supports network dynamics: SIS, neuronal, biomass.")
            if graph_file is None:
                raise ValueError("graph_file is required for DiffSTG.")
            if cached_pred_future_list is not None:
                pred_future_list = cached_pred_future_list
                if model_save_file is not None and load_model_when_cached:
                    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model, loaded_net_param = load_model_from_dir(
                        model_save_file,
                        device=device,
                        infer_params=infer_params,
                        method_config=method_config,
                    )
            else:
                if model_save_file is None:
                    raise ValueError("model_save_file is required when cache_path does not exist or force_recompute=True.")
                if infer_params is None:
                    infer_params = {"parallel_sampling": 10, "sequential_sampling": 1,"n_z_samples": 10, "diffusion_steps":20}
                device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, loaded_net_param = load_model_from_dir(
                    model_save_file,
                    device=device,
                    infer_params=infer_params,
                    method_config=method_config,
                )
                graph_data = load_diffstg_graph(graph_file)
                pred_future_list = run_diffstg_evaluation_cache(
                    model=model,
                    timeseries_datas=timeseries_datas,
                    pred_len=pred_len,
                    graph_data=graph_data,
                    cache_path=cache_path,
                    device=device,
                    force_recompute=force_recompute,
                    max_windows=max_windows,
                )
        elif cached_pred_future_list is not None:
            pred_future_list = cached_pred_future_list
            if model_save_file is not None and load_model_when_cached:
                device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, loaded_net_param = load_model_from_dir(
                    model_save_file,
                    device=device,
                    infer_params=infer_params,
                    method_config=method_config,
                )
        else:
            if model_save_file is None:
                raise ValueError("model_save_file is required when cache_path does not exist or force_recompute=True.")
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, loaded_net_param = load_model_from_dir(
                model_save_file,
                device=device,
                infer_params=infer_params,
                method_config=method_config,
            )
            pred_future_list = run_evaluation_cache(
                model=model,
                timeseries_datas=timeseries_datas,
                pred_len=pred_len,
                cache_path=cache_path,
                device=device,
                force_recompute=force_recompute,
                max_windows=max_windows,
            )
        pred_mean_list, uncertainty_ews_list = summarize_pred_future_list(pred_future_list, model=model)
    else:
        pred_future_list = None
        pred_mean_list = []
        uncertainty_ews_list = []

    valid_len = len(uncertainty_ews_list)
    result = {
        "pred_future_list": pred_future_list,
        "pred_mean": pred_mean_list,
        "ews": uncertainty_ews_list,
        "time_points": time_points[:valid_len],
        "cache_path": str(cache_path),
        "figure_path": str(resolve_figure_path(cache_path)),
        "torch_time_series": torch_time_series,
        "time_data": torch.as_tensor(time_data).cpu().detach().numpy(),
        "dynamic_type": dynamic_type,
        "sampling_t": sampling_t,
        "sample_window_step": sample_window_step,
        "windows": windows,
        "pred_len": pred_len,
        "task_model": task_model,
        "uncertainty_method": uncertainty_method,
        "uncertainty_source": "sampling" if need_sampling_uncertainty else None,
        "graph_file": str(_resolve_project_path(graph_file)) if graph_file is not None else None,
        "model_save_file": str(model_save_file) if model_save_file is not None else None,
        "loaded_net_param": loaded_net_param,
    }

    has_nsdiff_g_model = model is not None and hasattr(model, "cond_pred_model_g") and model.cond_pred_model_g is not None
    should_handle_nsdiff_g = need_gx_uncertainty and ("NsDiff" in str(task_model) or has_nsdiff_g_model)
    g_list = None
    if should_handle_nsdiff_g:
        if cached_nsdiff_g_list is not None:
            g_list = cached_nsdiff_g_list
        else:
            if model is None and model_save_file is not None:
                device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, loaded_net_param = load_model_from_dir(
                    model_save_file,
                    device=device,
                    infer_params=infer_params,
                    method_config=method_config,
                )
                result["loaded_net_param"] = loaded_net_param
            if model is not None and hasattr(model, "cond_pred_model_g") and model.cond_pred_model_g is not None:
                g_list = run_nsdiff_g_cache(
                    model=model,
                    timeseries_datas=timeseries_datas,
                    cache_path=nsdiff_path,
                    device=device,
                    pred_dim=pred_dim,
                    force_recompute=force_recompute,
                    max_windows=max_windows,
                )
            else:
                g_list = None
        if g_list is not None:
            g_pred_mean, g_ews = summarize_nsdiff_g_list(g_list, pred_dim=pred_dim)
            result["nsdiff_g"] = {
                "pred_future_list": g_list,
                "pred_mean": g_pred_mean,
                "ews": g_ews,
                "time_points": time_points[: len(g_ews)],
                "cache_path": str(nsdiff_path),
            }
            if uncertainty_method == "gx":
                result["pred_future_list"] = None
                result["pred_mean"] = g_pred_mean
                result["ews"] = g_ews
                result["time_points"] = time_points[: len(g_ews)]
                result["cache_path"] = str(nsdiff_path)
                result["figure_path"] = str(resolve_figure_path(nsdiff_path))
                result["uncertainty_source"] = "gx"

    if uncertainty_method == "gx" and g_list is None:
        raise ValueError(
            "uncertainty_method='gx' requires a task_model containing 'NsDiff' "
            "and a loaded model with cond_pred_model_g, or an existing _gx cache."
        )

    return result


def plot_single_model_check(result, pred_dim=0, title=None, save_path=None, axs=None):
    import matplotlib.pyplot as plt

    owns_figure = axs is None
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(6.2, 4.2),
                              #  sharex=False, gridspec_kw={"hspace": 0.08}
                                gridspec_kw={'hspace': 0.00}
                                )
    else:
        axs = np.asarray(axs, dtype=object).ravel()
        if len(axs) != 2:
            raise ValueError("plot_single_model_check expects 2 axes.")
        fig = axs[0].figure

    dynamic_type = result.get("dynamic_type")
    series = torch.as_tensor(result["torch_time_series"]).detach().cpu()
    time_data = np.asarray(result["time_data"])
    if dynamic_type in NETWORK_DYNAMICS:
        y = series[:, :, 0].mean(dim=0).numpy()
    else:
        y = series[0, :, pred_dim].numpy()

    axs[0].plot(time_data[: len(y)], y, color="#2F5597", linewidth=1.4)
    axs[0].set_ylabel("State")
    if title:
        axs[0].set_title(title)

    axs[1].plot(result["time_points"][: len(result["ews"])], result["ews"], ".", color="#C44E52", markersize=3)
    axs[1].set_ylabel("Uncertainty")
    axs[1].set_xlabel("Time")
    axs[1].sharex(axs[0])

    for ax in axs:
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.yaxis.label.set_size(10)
        ax.xaxis.label.set_size(10)

    if owns_figure:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def main():
    DEFAULT_RUN = {
        "model_save_file": "ews_results/model_compare/NsDiff/SIS",
        "data_file": "dataset/spdata_sde_SIS/barabasi_albert_30_0/SIS_dynamic_eta0.0001d0.5_increase.pt",
        "dynamic_type": "SIS",
        "task_model": None,
        "graph_file": "dataset/test_graph/barabasi_albert_30_0.graphml",
        "cache_path": None,
        "sample_window_step": None,
        "sampling_t": None,
        "pred_dim": 0,
        "force_recompute": False,
        "uncertainty_method": "gx",#"sampling"  # 多条预测轨迹的方差不确定性; "gx" # 直接使用 NsDiff 的 cond_pred_model_g / _gx 缓存; "both" # 两者都读取或生成
        "device": None,
        "infer_params":{"parallel_sampling": 50, "sequential_sampling": 1,"n_z_samples": 100, "diffusion_steps":20}
    }
    run_config = dict(DEFAULT_RUN)
    if run_config["model_save_file"] is None or run_config["data_file"] is None:
        print("Set DEFAULT_RUN['model_save_file'] and DEFAULT_RUN['data_file'] before running this file directly.")
        return
    result = uncertainty_ews(**run_config)
    print("cache_path:", result["cache_path"])
    print("figure_path:", result["figure_path"])
    print("num_windows:", len(result["ews"]))
    plot_single_model_check(
        result,
        title=str(run_config.get("dynamic_type") or "model check"),
        save_path=result["figure_path"],
    )


if __name__ == "__main__":
    main()
