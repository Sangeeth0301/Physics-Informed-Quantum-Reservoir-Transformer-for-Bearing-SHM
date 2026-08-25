# Changelog

All notable changes to the Physics-Informed Quantum Reservoir Transformer project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- IMS (NASA Run-to-Failure) dataset integration
- XJTU-SY variable-speed dataset support
- Phase 4: Dynamical Consistency Network (DCN) completion
- Hardware QPU backend support via PennyLane
- Comprehensive pytest test suite

---

## [1.0.0] – Phase 3 Complete — 2025

### Added
- **Phase 1: Signal Conditioning** — Butterworth bandpass (2–6 kHz), Hilbert envelope, windowing (2048/512 samples)
- **Phase 2: Multi-Resolution DMD** — Hankelization, Koopman mode extraction, spectral radius drift, eigenvalue analysis
- **Phase 3: Projected Quantum Kernel Reservoir (PQKR)** — 5-qubit angle encoding, CNOT ring entanglement, quantum fidelity kernel
- **Quantum SVM Readout** — One-Class SVM on Hilbert-space features producing Quantum Divergence Score
- **Temporal Transformer Encoder** — 15-window unsupervised multi-head attention for temporal pattern recognition
- **Phase 3 Statistical Hardening** — 10-seed robustness testing, KS-tests on eigenvalue distributions, UMAP clustering
- **Publication-grade figures** — ROC curves, phase portraits, UMAP embeddings, Master SI curves in `results/`
- Full CWRU benchmark dataset pipeline (8 classes, healthy vs. 7-mil incipient fault)

### Repository Setup
- Professional directory structure (`src/`, `scripts/`, `docs/`, `data/`, `results/`)
- `requirements.txt` with pinned dependencies (PyTorch 2.10, PennyLane 0.44, PyDMD 2025.8.1)
- GitHub Actions CI/CD pipeline
- MIT License, CITATION.cff, CONTRIBUTING.md

---

## [0.3.0] – Phase 3 PQKR Implementation

### Added
- `src/quantum/pqkr.py` — Projected Quantum Kernel Reservoir circuit
- `src/quantum/readout.py` — Quantum SVM readout layer
- `src/quantum/metrics.py` — Quantum divergence metrics
- `scripts/03_pqkr_analysis.py` — Full PQKR analysis pipeline
- `scripts/07_phase3_hardening.py` — Statistical robustness hardening
- `scripts/08_phase3_final_robustness.py` — Final robustness validation

---

## [0.2.0] – Phase 2 mrDMD Analysis

### Added
- `scripts/02_mrdmd_analysis.py` — Multi-resolution DMD analysis
- Koopman eigenvalue extraction and spectral radius tracking
- Healthy vs. fault comparison plots

---

## [0.1.0] – Phase 1 Data Pipeline

### Added
- `scripts/01_load_cwru_and_plot.py` — CWRU data loading (scipy + h5py fallback)
- Signal preprocessing: bandpass filter, Hilbert envelope, windowing
- `data/processed/` numpy arrays for downstream use
- Initial `README.md` and project structure
