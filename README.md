# PR-SAM: Prior-Refined Segment Anything Model for Quality-Robust CBCT Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of the paper: **"PR-SAM: Prior-Refined Segment Anything Model for Quality-Robust CBCT Segmentation in Adaptive Radiotherapy"**.

PR-SAM is a novel framework that conditions a **frozen Segment Anything Model (SAM)** with patient-specific planning contours to achieve robust segmentation on daily CBCT images, even under severe artifact conditions (e.g., sparse-view acquisitions).

## 📋 Abstract

Online adaptive radiotherapy (ART) requires fast, reliable segmentation of same-day CBCT despite artifacts that degrade image quality. We propose **PR-SAM**, a planning-CT guided approach that injects patient-specific anatomical priors into a frozen SAM backbone. By leveraging a lightweight **Prior Encoder**, a **CBCT Domain Adapter**, and an **Adaptive Correction Network**, PR-SAM reconciles prior knowledge with daily image evidence without fine-tuning the heavy SAM weights.

### Key Results (on CBCTLiTS dataset)
*   **Overall Performance**: Dice **0.9611** | HD95 **2.46 mm**
*   **Quality-Robustness**: Dice varies only **0.0005** across 5 quality tiers (32 to 490 projections).
*   **State-of-the-Art Comparison**:
    *   **+6.6%** Dice improvement over the best baseline (PolarUNet).
    *   **-86.8%** reduction in HD95 errors compared to PolarUNet.
    *   Significantly outperforms MedSAM and nnU-Net in sparse-view regimes.

## 🏗️ Architecture

PR-SAM consists of four key components designed for parameter efficiency (~2.3M trainable params):

1.  **Prior Encoder**: Converts registered pCT masks into multi-channel geometric features (Mask, SDF, Boundary).
2.  **CBCT Domain Adapter**: Adapts SAM's image embedding for CBCT texture without altering frozen weights.
3.  **Adaptive Correction Network**: Learns residual feature refinements to correct prior-image discrepancies.
4.  **Dual-Branch Decoding**: Fuses a *Prior-Aware* branch and an *Appearance-Only* branch for robust inference.

```mermaid
graph LR
    A[pCT Mask] --> B(Prior Encoder)
    C[CBCT Image] --> D(Frozen SAM Encoder)
    D --> E(CBCT Adapter)
    B --> F{Fusion & Correction}
    E --> F
    F --> G[Prior Branch]
    E --> H[Appearance Branch]
    G --> I(Output)
    H --> I
