from pathlib import Path

import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_ROOT = PROJECT_ROOT / "dataset" / "real_data"
OUT_DIR = PROJECT_ROOT / "paper_figures" / "outputs"


EXAMPLE_FILES = [
    REAL_DATA_ROOT / "bury_2021_anoxia" / "bury_2021_anoxia_tsid_1.pt",
    REAL_DATA_ROOT
    / "bury_2021_paleoclimate"
    / "bury_2021_paleoclimate_tsid_1_end_of_greenhouse_earth.pt",
    REAL_DATA_ROOT / "bury_2021_thermoacoustic" / "bury_2021_thermoacoustic_tsid_1.pt",
]


def as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def transition_x(record):
    ts = record["ts_dynamic"].detach().cpu()
    if "transition_index" in record:
        index = int(record["transition_index"])
        index = max(0, min(index, len(ts) - 1))
        return as_float(ts[index])
    if "transition_time" in record:
        return float(record["transition_time"])
    if "transition_age" in record:
        age = float(record["transition_age"])
        ts_np = ts.numpy()
        return float(ts_np[abs(ts_np - age).argmin()])
    return None


def channel_labels(record):
    data_type = record.get("data_type", "")
    if data_type == "anoxia":
        return ["Mo", "U"]
    if data_type == "thermoacoustic":
        return ["Pressure"]
    if data_type == "paleoclimate":
        return ["Proxy"]
    return [f"dim {i}" for i in range(record["ys_dynamic"].shape[1])]


def plot_examples():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(EXAMPLE_FILES), 1, figsize=(9, 7.5), sharex=False)

    if len(EXAMPLE_FILES) == 1:
        axes = [axes]

    for ax, path in zip(axes, EXAMPLE_FILES):
        record = torch.load(path, map_location="cpu")
        ts = record["ts_dynamic"].detach().cpu().numpy()
        ys = record["ys_dynamic"].detach().cpu().numpy()
        labels = channel_labels(record)

        for dim in range(ys.shape[1]):
            ax.plot(ts, ys[:, dim], linewidth=1.2, label=labels[dim] if dim < len(labels) else f"dim {dim}")

        tx = transition_x(record)
        if tx is not None:
            ax.axvline(tx, color="crimson", linestyle="--", linewidth=1.5, label="transition")

        title = record.get("name", path.stem).replace("bury_2021_", "").replace("_", " ")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("value")
        ax.legend(loc="best", frameon=False, fontsize=8)

    axes[-1].set_xlabel("time axis in converted .pt")
    fig.tight_layout()
    out_path = OUT_DIR / "bury_real_transition_points_examples.png"
    fig.savefig(out_path, dpi=300)
    print(out_path)


if __name__ == "__main__":
    plot_examples()
