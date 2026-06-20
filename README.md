# Methodology

Implementation code for the methodology chapter of the thesis on federated learning under non-IID label skew.

Repository: https://github.com/mohmdumer/FedDiffkd.git

This repository contains the scripts used to run the three main experiments in the paper:

- E1: Baseline FedAvg
- E2: FedAvg + server-side knowledge distillation with DDIM-generated samples
- E3: FedAvg + server-side KD + per-class FID quality filtering

The code is organized so the baseline, KD-only, and KD+FID variants can be run separately and compared under the same CIFAR-10 / Dirichlet $\alpha = 0.6$ setting.

## Repository Layout

```text
methodology/
├── E1_baseline.py              # Baseline FedAvg experiment
├── E2_fedavg_kd.py             # FedAvg + server-side KD experiment
├── E3_fedavg_kd_fid.py         # FedAvg + KD + FID filtering experiment
├── fl_with_kd.py               # Main FL pipeline with KD
├── server_kd.py                # Server-side KD helper functions
├── xai_analysis.py             # Grad-CAM / XAI analysis
└── README.md                   # Project overview and usage
```

Related files used with this folder:

- `ddim_sampler.py` for fast DDIM generation
- `per_class_fid.py` for per-class FID computation
- `data/` for CIFAR-10 downloads and local cache
- `results/` and `checkpoints/` for outputs produced during experiments

## Method Summary

Each federated round follows the same high-level flow:

1. Sample a subset of clients.
2. Train each selected client locally on its private CIFAR-10 shard.
3. Aggregate client updates with FedAvg.
4. On the server, generate synthetic samples with DDIM.
5. Distill client knowledge into the global model using weighted ensemble teacher predictions.
6. Optionally apply per-class FID-based quality filtering.
7. Evaluate the global model and save checkpoints/results.

The repository also includes XAI analysis utilities to generate Grad-CAM visualizations for the final model.

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- pandas
- numpy
- scikit-learn
- a CUDA-capable GPU is recommended for the KD and DDIM experiments

If you are running the notebooks/scripts on Kaggle, make sure the DDPM checkpoint path points to the uploaded dataset location.

## How to Run

### E1: Baseline FedAvg

Run `E1_baseline.py` to reproduce the federated averaging baseline without synthetic augmentation or knowledge distillation.

### E2: FedAvg + KD

Run `E2_fedavg_kd.py` for the main method with server-side KD and DDIM-generated synthetic samples.

### E3: FedAvg + KD + FID

Run `E3_fedavg_kd_fid.py` for the enhanced variant that adds per-class FID weighting.

### XAI Analysis

After training, run `xai_analysis.py` on the saved checkpoint to generate Grad-CAM visualizations and related interpretability outputs.

## Outputs

Typical outputs are written to:

- `checkpoints/` for best or latest model weights
- `results/` for CSV and JSON summaries
- `plots/` for figures if the script creates them

## Notes

- The code is written to support the thesis methodology experiments, not a production FL deployment.
- Large generated artifacts such as datasets, checkpoints, and local virtual environments should normally be excluded from version control.
- Update any hard-coded checkpoint paths before running in a new environment.

## Citation

If you use this code in your work, please cite the thesis or the associated paper.

## License

Add a license file before publishing the repository publicly.
