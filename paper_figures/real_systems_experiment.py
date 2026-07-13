"""Paper real-data figure: compose multiple real-data EWS subfigures."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch

from common import OUTPUT_ROOT, ensure_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_and_analysis import real_data_analysis as rda  # noqa: E402


DEFAULT_REAL_DATA_NAMES = (
    "bury_2021_anoxia_tsid_3",
    "bury_2021_anoxia_tsid_6",
    "bury_2021_anoxia_tsid_9",
# "bury_2021_anoxia_tsid_1",
#     "bury_2021_anoxia_tsid_2",
#     "bury_2021_anoxia_tsid_4",
# "bury_2021_anoxia_tsid_5",
#     "bury_2021_anoxia_tsid_7",
#     "bury_2021_anoxia_tsid_8",
# "bury_2021_anoxia_tsid_10",
#     "bury_2021_anoxia_tsid_11",
#     "bury_2021_anoxia_tsid_12",


)

SIGNAL_STYLE = {
    "model_uncertainty": ("#B64342", ".", "MPV"),
    "model_trend": ("#0F4D92", "-.", "MPV Trend"),
    "ar1": ("#E28E2C", "-", "AR(1)"),
    "variance": ("#42949E", "-", "Variance"),
    "sample-entropy-1": ("#9A4D8E", "-", "Sample Entropy"),
}


def apply_layout_scale(args, scale: float) -> None:
    label_size = max(4.8, args.label_fontsize * scale)
    tick_size = max(4.5, args.tick_fontsize * scale)
    title_size = max(5.2, args.title_fontsize * scale)
    panel_label_size = max(6.0, args.panel_label_fontsize * scale)
    args._label_fontsize = label_size
    args._tick_fontsize = tick_size
    args._title_fontsize = title_size
    args._panel_label_fontsize = panel_label_size
    args._line_width = max(0.55, args.line_width * scale)
    args._marker_size = max(1.0, args.marker_size * scale)
    mpl.rcParams.update({"font.size": label_size})


def resolve_grid_and_size(n_panels: int, n_axes_per_panel: int, args) -> tuple[int, int, float, float]:
    if args.ncols is not None:
        if args.ncols < 1:
            raise ValueError("--ncols must be at least 1.")
        ncols = min(args.ncols, n_panels)
    else:
        ncols = min(args.max_cols, n_panels)
    nrows = math.ceil(n_panels / ncols)

    height_per_axis = args.height_per_axis
    if args.compact_main and n_panels <= 3:
        height_per_axis = min(height_per_axis, args.main_height_per_axis)
        args.panel_hspace = min(args.panel_hspace, args.main_panel_hspace)

    base_width = args.width_per_col * ncols
    base_height = height_per_axis * n_axes_per_panel * nrows
    fig_width, fig_height = base_width, base_height

    if args.fit_page:
        fig_width = min(base_width, args.max_fig_width)
        fig_height = min(base_height, args.max_fig_height)

    width_scale = fig_width / base_width if base_width else 1.0
    height_scale = fig_height / base_height if base_height else 1.0
    apply_layout_scale(args, min(width_scale, height_scale, 1.0))
    return ncols, nrows, fig_width, fig_height


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )


def infer_tsid(data_name: str, record: dict) -> int | None:
    if "tsid" in record:
        return int(record["tsid"])
    match = re.search(r"_tsid_(\d+)", data_name)
    return int(match.group(1)) if match else None


def anoxia_title_lookup(data_root: Path) -> dict[int, str]:
    csv_path = Path(data_root) / "bury_2021_anoxia" / "data_transitions.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    required = {"tsid", "ID", "Core"}
    if not required.issubset(df.columns):
        return {}
    lookup = {}
    for tsid, group in df.groupby("tsid"):
        transition_id = str(group["ID"].iloc[0])
        core = str(group["Core"].iloc[0])
        lookup[int(tsid)] = f"{transition_id}-{core}"
    return lookup


def display_title_for_record(data_name: str, record: dict, data_root: Path) -> str:
    if str(record.get("data_type", "")).lower() == "anoxia" or data_name.startswith("bury_2021_anoxia"):
        tsid = infer_tsid(data_name, record)
        lookup = anoxia_title_lookup(data_root)
        if tsid in lookup:
            return lookup[tsid]
    return data_name


def load_dataset_panel(data_name_or_path: str, args):
    data_path = rda.resolve_real_data_path(data_name_or_path, args.data_root)
    record = torch.load(data_path, map_location="cpu")
    data_name = data_path.stem
    display_title = display_title_for_record(data_name, record, args.data_root)
    ts, y, signal_data, signal_table = rda.build_signal_data(record, data_name, args)
    transition_time = rda.get_transition_time_from_record(record)
    return data_name, display_title, ts, y, signal_data, signal_table, transition_time


def plot_dataset_subfigure(subfig, panel_label: str, display_title: str, ts, y, signal_data, transition_time, args):
    axes = subfig.subplots(
        1 + len(args.signals),
        1,
        sharex=True,
        gridspec_kw={"hspace": args.panel_hspace, "height_ratios": [1.25] + [1.0] * len(args.signals)},
    )
    if hasattr(axes, "flat"):
        axes = list(axes.flat)
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]

    axes[0].plot(ts, y, color="#0F4D92", linewidth=args._line_width)
    axes[0].set_ylabel(args.trajectory_ylabel)
    axes[0].set_title(display_title, fontsize=args._title_fontsize, pad=2.5)

    for axis_index, signal in enumerate(args.signals, start=1):
        ax = axes[axis_index]
        if signal == "bury_ml":
            bury_df = signal_data[signal]
            colors = {
                "fold_prob": "#0F4D92",
                "hopf_prob": "#E28E2C",
                "branch_prob": "#42949E",
                "null_prob": "#767676",
            }
            labels = {
                "fold_prob": "fold",
                "hopf_prob": "Hopf",
                "branch_prob": "branch",
                "null_prob": "null",
            }
            for col in rda.BURY_ML_PROB_COLUMNS:
                ax.plot(bury_df["time"], bury_df[col], color=colors[col], linewidth=args._line_width, label=labels[col])
            ax.set_ylim(-0.03, 1.03)
            ax.set_ylabel("Bury DL")
            if not args.hide_bury_legend:
                ax.legend(
                    loc="upper left",
                    ncol=2,
                    fontsize=max(4.5, args._tick_fontsize),
                    handlelength=1.1,
                    columnspacing=0.55,
                    labelspacing=0.2,
                )
        else:
            times, values = signal_data[signal]
            color, line_style, label = SIGNAL_STYLE[signal]
            ax.plot(
                times,
                values,
                linestyle="None" if line_style == "." else line_style,
                marker="." if line_style == "." else None,
                color=color,
                linewidth=args._line_width,
                markersize=args._marker_size,
            )
            ax.set_ylabel(label)

    if transition_time is not None:
        for ax in axes:
            ax.axvline(transition_time, color="#B64342", linestyle="--", linewidth=max(0.5, args._line_width), alpha=0.72)

    for ax in axes:
        ax.grid(alpha=0.16, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=args._tick_fontsize, length=2.0, width=0.6)
        ax.yaxis.label.set_size(args._label_fontsize)
        ax.set_xlim(float(min(ts)), float(max(ts)))

    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.label.set_size(args._label_fontsize)
    subfig.text(0.015, 0.99, panel_label, ha="left", va="top", fontsize=args._panel_label_fontsize, fontweight="bold")


def build_fig_real(args) -> None:
    set_publication_style()
    rda.validate_signals(args.signals)
    output_dir = ensure_output(args.output_dir)

    panels = [load_dataset_panel(data_name, args) for data_name in args.data_real_names]
    n_panels = len(panels)
    n_axes_per_panel = 1 + len(args.signals)
    ncols, nrows, fig_width, fig_height = resolve_grid_and_size(n_panels, n_axes_per_panel, args)

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    subfigs = fig.subfigures(nrows, ncols, squeeze=False, wspace=args.subfig_wspace, hspace=args.subfig_hspace)
    signal_tables = []

    for index, panel in enumerate(panels):
        row, col = divmod(index, ncols)
        label = chr(ord("a") + index)
        data_name, display_title, ts, y, signal_data, signal_table, transition_time = panel
        plot_dataset_subfigure(subfigs[row][col], label, display_title, ts, y, signal_data, transition_time, args)
        signal_tables.append(signal_table)

    for index in range(n_panels, nrows * ncols):
        row, col = divmod(index, ncols)
        subfigs[row][col].set_visible(False)

    output_stem = output_dir / args.output_name
    #fig.savefig(f"{output_stem}.svg")
    fig.savefig(f"{output_stem}.pdf")
    fig.savefig(f"{output_stem}.png", dpi=args.dpi)
    plt.close(fig)

    if not args.no_save_csv and signal_tables:
        import pandas as pd

        pd.concat(signal_tables, ignore_index=True).to_csv(f"{output_stem}_signals.csv", index=False)
    #print(f"saved: {output_stem}.svg")
    print(f"saved: {output_stem}.pdf")
    print(f"saved: {output_stem}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "dataset" / "real_data")
    parser.add_argument("--real-data-result-root", type=Path, default=PROJECT_ROOT / "ews_results" / "real_data")
    parser.add_argument("--bury-prob-root", type=Path, default=PROJECT_ROOT / "ews_results" / "bury_2021_ml_probs")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-name", default="fig_real")
    parser.add_argument("--data-real-names", nargs="+", default=list(DEFAULT_REAL_DATA_NAMES))
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["model_uncertainty",
                # "model_trend",
                 "ar1", "variance", "sample-entropy-1", "bury_ml"],
        help=f"Signals after each trajectory panel. Valid: {sorted(rda.VALID_SIGNALS)}",
    )
    parser.add_argument("--ncols", type=int, default=None, help="Fixed number of dataset panels per row.")
    parser.add_argument("--max-cols", type=int, default=3)
    parser.add_argument("--pred-dim", type=int, default=0)
    parser.add_argument("--model-key", choices=sorted(rda.MODEL_KEY_DIRS), default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--model-input", choices=["raw", "detrended"], default="detrended")
    parser.add_argument("--model-detrend-method", choices=["Lowess", "Gaussian", "none"], default="Gaussian")
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
    parser.add_argument("--trajectory-ylabel", default="Mo")
    parser.add_argument("--width-per-col", type=float, default=3.6)
    parser.add_argument("--height-per-axis", type=float, default=1.05)
    parser.add_argument("--compact-main", dest="compact_main", action="store_true", default=True)
    parser.add_argument("--no-compact-main", dest="compact_main", action="store_false")
    parser.add_argument("--main-height-per-axis", type=float, default=0.90)
    parser.add_argument("--main-panel-hspace", type=float, default=0.025)
    parser.add_argument("--fit-page", dest="fit_page", action="store_true", default=True)
    parser.add_argument("--no-fit-page", dest="fit_page", action="store_false")
    parser.add_argument("--max-fig-width", type=float, default=7.2, help="Maximum figure width in inches.")
    parser.add_argument("--max-fig-height", type=float, default=9.2, help="Maximum figure height in inches.")
    parser.add_argument("--label-fontsize", type=float, default=7.2)
    parser.add_argument("--tick-fontsize", type=float, default=6.2)
    parser.add_argument("--title-fontsize", type=float, default=8.0)
    parser.add_argument("--panel-label-fontsize", type=float, default=9.6)
    parser.add_argument("--line-width", type=float, default=0.95)
    parser.add_argument("--marker-size", type=float, default=2.2)
    parser.add_argument("--panel-hspace", type=float, default=0.04)
    parser.add_argument("--subfig-wspace", type=float, default=0.025)
    parser.add_argument("--subfig-hspace", type=float, default=0.035)
    parser.add_argument("--hide-bury-legend", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--no-save-csv", action="store_true")
    args = parser.parse_args()
    build_fig_real(args)


if __name__ == "__main__":
    main()
