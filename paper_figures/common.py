from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EWS_ROOT = PROJECT_ROOT / "ews_results"
Source_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "paper_figures" / "outputs"

DATASETS = ("biomass", "neuronal", "SIS")
TRENDS = ("decrease", "increase")
GRAPH_TYPES = ("BA", "ER", "WS")
MODEL_COMPARE_MODELS = ("NsDiff", "DiffSTG")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ews-root", type=Path, default=EWS_ROOT)
    parser.add_argument("--source-root", type=Path, default=Source_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--trend", choices=TRENDS, default="increase")


def select_one_or_all(value: str | None, choices: tuple[str, ...]) -> list[str]:
    return [value] if value else list(choices)


def as_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def dynamics_filename(dataset_type: str, data_trend: str) -> str:
    if dataset_type == "biomass":
        return f"biomass_dynamic_eta0.005r0.7_{data_trend}.pt"
    if dataset_type == "neuronal":
        return f"neuronal_dynamic_eta0.01tau2.0_{data_trend}.pt"
    if dataset_type == "SIS":
        return f"SIS_dynamic_eta0.0001d0.5_{data_trend}.pt"
    raise ValueError(f"unknown dataset_type: {dataset_type}")


def graph_name(graph_type: str) -> str:
    if graph_type == "BA":
        return "barabasi_albert_30_0"
    if graph_type == "ER":
        return "erdos_renyi_50_0"
    if graph_type == "WS":
        return "small-world_70_0"
    raise ValueError(f"unknown graph_type: {graph_type}")


def spdata_source_path(source_root: Path, dataset_type: str, graph: str, data_name: str) -> Path:
    return source_root / f"spdata_sde_{dataset_type}" / graph / data_name


def slbp_source_path(source_root: Path, total_time: str, data_trend: str, d_value: str, test_data: bool = False) -> Path:
    folder = "SLBP_model_data_test" if test_data else "SLBP_model_data"
    return source_root / folder / f"SLBP_dynamic_total_time_{total_time}_N_{data_trend}" / f"SLBP_dynamic_D_{d_value}.pt"


def load_dynamic_record(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    record = torch.load(path, map_location="cpu")
    if not isinstance(record, dict) or "ts_dynamic" not in record or "ys_dynamic" not in record:
        raise ValueError(
            f"{path} must be a torch-saved dict with keys 'ts_dynamic' and 'ys_dynamic'."
        )
    return record["ts_dynamic"], record["ys_dynamic"]


def ensure_output(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dynamics_title(dataset_type: str) -> str:
    if dataset_type == "biomass":
        return "Resource biomass"
    if dataset_type == "neuronal":
        return "Wilson-Cowan neuronal"
    if dataset_type == "SIS":
        return "SIS"
    raise ValueError(f"unknown dataset_type: {dataset_type}")


def parameters(dataset_type: str) -> tuple[int, float]:
    if dataset_type in ("biomass", "neuronal"):
        return 5, 10
    if dataset_type == "SIS":
        return 20, 0.1
    raise ValueError(f"unknown dataset_type: {dataset_type}")
