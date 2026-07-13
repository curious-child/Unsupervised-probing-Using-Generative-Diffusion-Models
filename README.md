# Unsupervised Probing Critical Transitions in Complex Systems Using Generative Diffusion Models

This repository contains the code, model configurations, trained checkpoints, simulation programs, and figure-generation workflows for the manuscript **Unsupervised Probing Critical Transitions in Complex Systems Using Generative Diffusion Models** by Peng Zhang, Jun Fu, and Rolf Findeisen.

The study treats diffusion models as conditional generative forecasters and examines whether their predictive uncertainty reflects changes near critical transitions. At inference, each model generates multiple plausible future trajectories from a rolling observation window. Their dispersion is summarized as the mean predictive variance (MPV). Across several model architectures and dynamical systems, NsDiff produces a particularly consistent localized collapse in predictive uncertainty.

## Models and systems

The repository includes four diffusion-model implementations:

- **NsDiff**: a non-stationary diffusion model with an explicit learned variance pathway.
- **DiffSTG**: a graph-based diffusion model for spatiotemporal forecasting.
- **DiffusionTS**: a transformer-based time-series diffusion model.
- **TMDM**: a conditional diffusion model for multivariate time series.

Experiments cover three networked dynamical systems and one controlled single-node system:

- Resource biomass dynamics
- Wilson-Cowan neuronal dynamics
- Susceptible-infected-susceptible (SIS) epidemic dynamics
- Shallow Lake with Bream and Pike (SLBP) dynamics

The supplied graph topologies include Barabasi-Albert, Erdos-Renyi, and small-world networks.

## Repository structure

```text
configs/                  Training and grid-search YAML configurations
dataset/                  Simulation programs and reusable graph topologies
evaluation_and_analysis/  Unified model inference, cache processing, and EWS analysis
ews_results/              Trained model parameters and model configurations
loss_functions/           Training objectives
models/                   NsDiff, DiffSTG, DiffusionTS, and TMDM implementations
optimizers/               Optimizer and scheduler factories
paper_figures/            Manuscript experiment and figure entry points
train/                    Training loops
utils/                    Data preparation, model loading, and common utilities
```

Large simulated trajectories and generated `.pt` prediction caches are intentionally not versioned. The trajectories can be regenerated with the programs in `dataset/`; the complete archived datasets will be released separately through Zenodo.

## Environment

The reference environment is Python 3.10.11 with PyTorch 2.0.1, CUDA 11.8, and PyTorch Geometric 2.5.3. From the repository root:

```bash
conda create -n torch_gym python=3.10.11 -y
conda activate torch_gym
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The provided requirements target CUDA 11.8. For CPU-only execution or another CUDA version, install the matching PyTorch and PyTorch Geometric wheels first, then install the remaining packages.

## Generate simulation data

Generate or update network topologies with:

```bash
python dataset/graph_generate.py
```

The dynamical simulation entry points are:

```bash
python dataset/spdata_sde_biomass_dynamic_gene.py
python dataset/spdata_sde_neuronal_dynamic_gene.py
python dataset/spdata_sde_SIS_dynamic_gene.py
python dataset/SLBP_dynamic_gene.py
```

The corresponding `*_contant.py` programs generate trajectories under constant control-parameter settings. Dataset ranges, noise levels, graph paths, and output directories are configured near the executable section of each simulation program.

## Train models

Networked-system models are trained with:

```bash
python main_SSLtrain_diffusion_spdata.py \
  --cfg configs/grid_search/diffusion_model_NsDiff.yaml \
  --train_mode hold_out \
  --repeat 1
```

Single-node or multivariate time-series models use:

```bash
python main_SSLtrain_diffusion_timeseries.py \
  --cfg configs/grid_search/diffusion_model_NsDiff.yaml \
  --train_mode hold_out \
  --repeat 1
```

Alternative model configurations are available as `diffusion_model_DiffSTG.yaml`, `diffusion_model_DiffusionTS.yaml`, and `diffusion_model_TMDM.yaml`. Device selection, input paths, window length, prediction horizon, sampling interval, and optimizer settings are defined in the YAML files.

## Model inference and MPV analysis

`evaluation_and_analysis/diffusion_model_uncertainy.py` is the shared inference and cache-processing entry point. It supports network data in `[Node, Time, Feature]` form, SLBP time series, DiffSTG graph inputs, sampling-based MPV, and the faster NsDiff `gx` variance estimate. Existing caches are read first; missing caches can be regenerated from the corresponding checkpoint and configuration.

For a direct single-case check, edit `DEFAULT_RUN` in the script's `main()` function and run:

```bash
python evaluation_and_analysis/diffusion_model_uncertainy.py
```

## Reproduce manuscript figures

Run figure scripts from the repository root. Outputs are written to `paper_figures/outputs/` as PNG and PDF files.

| Figure | Entry point | Experiment |
| --- | --- | --- |
| Fig. 1 | `paper_figures/diffusion mdoel compare_experiment.py` | Diffusion-model comparison across network dynamics |
| Fig. 2 | `paper_figures/graph_generalization_experiment.py` | Generalization across graph topologies |
| Fig. 3 | `paper_figures/dynamics_transfer_experiment.py` | Transfer across dynamical systems |
| Fig. 4 | `paper_figures/model_sensitivity_experiment.py` | Window and prediction-horizon sensitivity |
| Fig. 5 | `paper_figures/tipping_types_experiment.py` | Control-parameter rate and noise effects |
| Fig. 6 | `paper_figures/SLBP_model_analysis_experiment.py` | Uncertainty mechanism analysis |
| Fig. 7 | `paper_figures/model_train_analysis.py` | Training-regime and mechanism ablations |
| Fig. 8 | `paper_figures/train_source_comparison_experiment.py` | Training-data source comparison |
| Fig. 9 | `paper_figures/smoothing_experiment.py` | Smoothed-input false-collapse control |
| Real systems | `paper_figures/real_systems_experiment.py` | Empirical early-warning analysis |

For example:

```bash
python "paper_figures/diffusion mdoel compare_experiment.py" --trend increase
python paper_figures/graph_generalization_experiment.py --trend increase
python paper_figures/dynamics_transfer_experiment.py --trend decrease
python paper_figures/smoothing_experiment.py
```

Most scripts accept `--source-root`, `--ews-root`, and `--output-dir` so that datasets, checkpoints, and generated figures can be stored outside the repository.

## Data and checkpoints

- Reusable graph topology files and all simulation programs are included in this repository.
- Trained model parameters and their YAML configurations are included under `ews_results/`.
- Simulated trajectories, generated prediction caches, and large archives are excluded from Git.
- The complete simulation datasets and figure source data will be deposited on Zenodo. The DOI will be added here when the archive is released.

When using empirical datasets, follow the license and citation requirements of their original sources.

## Citation

Please cite the associated manuscript if this repository contributes to your work. Citation metadata are provided in [`CITATION.cff`](CITATION.cff) and will be updated with the final journal reference and DOI after publication.

## License

This software is released under the [MIT License](LICENSE).
