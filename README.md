# PR-SAM: Patient-Specific Prior-Refined Prompting of a Frozen SAM for Quality-Robust CBCT Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.9+](https://img.shields.io/badge/pytorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **PR-SAM** (Prior-Refined SAM), a novel framework for quality-robust CBCT segmentation in online adaptive radiotherapy.

<p align="center">
  <img src="results/figures/two_patients_comparison.png" width="100%" alt="Segmentation Comparison">
</p>

## 📋 Abstract

Online adaptive radiotherapy (ART) requires fast, reliable segmentation of same-day CBCT despite artifacts that degrade image quality. We propose **PR-SAM**, a planning-CT guided approach that injects patient-specific anatomical priors into a frozen Segment Anything Model (SAM) to achieve robust CBCT segmentation without backbone fine-tuning.

**Key Results:**
- **Dice: 0.9611** | **HD95: 2.46 mm** (overall performance)
- **Quality-invariant**: Dice range 0.0005 across all quality tiers
- **+4.9% Dice** improvement over best baseline (U-Net)
- **-75.6% HD95** reduction compared to best baseline

<p align="center">
  <img src="results/figures/quality_robustness_multi_metrics.png" width="100%" alt="Quality Robustness">
</p>

## 🏗️ Architecture

PR-SAM consists of four key components:

1. **Prior Encoder** (`N_prior`): Converts registered pCT masks into multi-channel representations (binary, signed-distance, boundary maps)
2. **Adaptive Correction Network** (`N_correct`): Learns feature-level residual refinement
3. **CBCT Domain Adapter**: Preserves domain-specific details from CBCT images
4. **Dual-Branch Decoding**: Combines prior-guided and correction branches for robust inference

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    PR-SAM Architecture                   │
                    └─────────────────────────────────────────────────────────┘
                    
    pCT Mask ──► Prior Encoder ──► Prior Features ──┬──► Prior Branch ──┐
                    (0.37M)              │          │                    │
                                         │          ▼                    │
                                         └──► Correction Network ──►─────┼──► Fusion ──► Output
                                              (1.84M)                    │
    CBCT ────► SAM Encoder ──► CBCT Features ──► CBCT Adapter ──────────┘
               (frozen 93.7M)                    (0.07M)
```

## 📁 Project Structure

```
PR-SAM/
├── models/
│   ├── proposed/
│   │   └── pgrsam/
│   │       ├── pgr_sam_model.py      # Main PR-SAM model
│   │       ├── pct_prior_encoder.py  # Prior encoder network
│   │       ├── correction_network.py # Adaptive correction network
│   │       └── ablation_models.py    # Ablation study variants
│   └── baselines/
│       ├── unet/                     # U-Net
│       ├── attention_unet/           # Attention U-Net
│       ├── nnunet/                   # nnU-Net
│       ├── resunet/                  # ResU-Net
│       ├── vnet/                     # V-Net
│       └── polar_unet/               # PolarUNet
├── data/
│   ├── dataset.py                    # Base dataset class
│   └── multi_quality_dataset.py      # Multi-quality CBCT dataset
├── configs/
│   ├── proposed_models.yaml          # PR-SAM configuration
│   ├── baseline_models.yaml          # Baseline configurations
│   └── ablation_study.yaml           # Ablation study configs
├── segment_anything_medsam/          # SAM backbone
├── utils/
│   ├── metrics.py                    # Evaluation metrics (Dice, IoU, HD95)
│   └── data_preprocessing.py         # Data preprocessing utilities
├── train_pgrsam.py                   # Training script for PR-SAM
├── train.py                          # Training script for baselines
├── evaluate_pgrsam.py                # Evaluation script
└── preprocess_dataset.py             # Data preprocessing



