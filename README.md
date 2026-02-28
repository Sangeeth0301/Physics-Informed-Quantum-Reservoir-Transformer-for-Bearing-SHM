# Physics-Informed Quantum Reservoir Transformer for Bearing SHM

**Status**: Research & Development  
**Goal**: Ultra-early bearing fault detection for predictive maintenance

This project aims to develop a novel unsupervised, hybrid quantum-classical framework specifically targeting the ultra-early detection of incipient bearing instability in rotating machinery. By diagnosing the birth of dynamical instability rather than waiting for traditional post-fault spectral frequencies (BPFO, BPFI, BSF), this research targets Industry 4.0 predictive maintenance via:
1. Multi-resolution Dynamic Mode Decomposition (mrDMD)
2. Koopman Operator Spectral Analysis
3. Projected Quantum Kernel Reservoir (PQKR) Feature Lifting
4. Physics-Guided Latent Output consistency

## Implementation Progress

The following modular phases are currently complete, fully verified, and hardened for publication reproducibility:

### Phase 1: Data Preparation & Signal Conditioning 
- Robust loading of bearing acceleration data (currently via CWRU `.mat` or `v7.3` hdf5 formats).
- Implementation of safe, research-grade signal conditioning (Butterworth bandpass, Hilbert Transform envelopes, empirical mean centering/scaling).
- Extracting sliding overlapping tensors (2048 samples, 512 stride).

### Phase 2: Classical Multi-Resolution Dynamics Analysis 
- Extraction of high-dimensional Hankel matrices via strict delay embedding.
- Deployment of multi-level standard/mrDMD factorization models.
- Statistical Koopman feature extraction including spectral radius mapping, instability node counts, and characteristic mean frequency tracing.

### Phase 3: Projected Quantum Kernel Reservoir (PQKR) 
- Leakage-safe Z-score normalization scaling and dimensional variance compression (via PCA) bottlenecked to scalable qubits architectures.
- Exact algorithmic mapping mapping a generic continuous feature space directly into a deterministic PennyLane parametrised rotation matrix (`AngleEmbedding`).
- Robust untrainable continuous quantum feature mapping evaluated over multi-seeded variance benchmarks against Gaussian RBF baselines via Frobenius divergence and theoretical maximum mean discrepancy (MMD).
- Comprehensive qubit ablation reporting (evaluating scaling power up to 6 qubits).

## Setup (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Verification Scripts
You can replicate our results by executing the Phase sequence:
```powershell
# Phase 1
python scripts/01_load_cwru_and_plot.py

# Phase 2
python scripts/02_mrdmd_analysis.py

# Phase 3
python scripts/03_pqkr_analysis.py

# Publish Graphics (Q1 Visualizations)
python scripts/04_q1_publication_graphics.py

# Publish Statistical Hardened Results (Multi-Seed & Annotations)
python scripts/05_q1_statistical_hardening.py
```
