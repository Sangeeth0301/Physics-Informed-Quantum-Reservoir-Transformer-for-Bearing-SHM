# Quantum-Enhanced Koopman Operator Learning for Incipient Bearing Instability Early Warning

**Status**: Research & Development (Phase 1-3 Completed)  
**Goal**: Ultra-early bearing fault detection for predictive maintenance

## 📖 Project Overview
We are building an **advanced hybrid quantum-classical system** that can detect **very early (incipient) faults** in rolling element bearings of rotating machines (motors, turbines, pumps) **much earlier** than traditional methods.

- **Input**: Vibration signals from accelerometers.
- **Core Innovation**: Detecting the **birth of dynamical instability** — subtle changes in the underlying dynamics (mode splitting, spectral radius drift) while the signal appears almost healthy, rather than waiting for obvious fault frequencies (BPFO, BPFI) to manifest.
- **Approach**: Train only on healthy data (unsupervised). We fuse physical rules of bearing motion, multi-scale dynamics analysis, and high-dimensional quantum embeddings to make microscopic instability signals stand out.
- **Output**: A continuous "Instability Score" (SI) that provides an early warning prior to spectral fault visibility.

## 🧠 The Core Problem
Traditional methods (FFT, classical ML, Deep Learning) fail at early detection because they:
- Rely heavily on faulty training data (rare in real, dynamic industrial settings).
- Look for frequency peaks *after* the fault has already physically grown.
- Ignore the nonlinear physical dynamics of the bearing (Hertzian contact, impulsive forcing).
- Lack sensitivity to the subtle dynamical instabilities that precede physical fracture.

**Our Solution:** Detect the loss of dynamical stability using only healthy data baselines, physical constraints, and exponential quantum sensitivity.

## 🚀 Architecture Pipeline
1. **Signal Conditioning**: Process raw vibration inputs (Envelope, Normalized Windows).
2. **Multi-Resolution DMD (mrDMD)**: Extract multi-scale modes and decay rates.
3. **Koopman Operator Estimation**: Map spectral drift and stability eigenvalues.
4. **Projected Quantum Kernel Reservoir (PQKR)**: Entangle features into a Quantum Hilbert Space to exponentiate sensitivity.
5. **Dynamical Consistency Network (DCN)** *(Upcoming)*: 1D-CNN Autoencoder learning healthy reconstruction.
6. **Physics-Guided Latent Evolution** *(Upcoming)*: Neural ODE enforcing nonlinear Jeffcott/Hertzian models.
7. **Change-Point Detection (SI Score)** *(Upcoming)*: Fusion of Recon Error + Koopman Drift + Physics Residual + Quantum Divergence.

## 📈 Current Progress
**Phases 1-3 are complete, fully verified, and statistically hardened for Q1 publication:**
- ✅ Environment setup & Dataset loading (CWRU benchmark).
- ✅ Preprocessing (Bandpass, Hilbert Envelope, Extracting 2048-sample overlapped tensors).
- ✅ mrDMD extraction & Koopman Operator feature tracking.
- ✅ PQKR quantum embedding & Classical RBF baseline comparison.
- ✅ Extensive Statistical Hardening: Multi-seed sensitivity, UMAP density tracking, Gaussian noise robustness, Cohen's *d* Effect Size separation bounds, and Kolmogorov-Smirnov Spectral log-decay tests.

## 💻 Setup & Installation
```powershell
# Create and activate environment
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## 🔄 Reproduction
Execute the following scripts sequentially to reproduce Phase 1-3 academic baselines and optimal Q1 results:
```powershell
python scripts/01_load_cwru_and_plot.py
python scripts/02_mrdmd_analysis.py
python scripts/07_phase3_hardening.py
python scripts/08_phase3_final_robustness.py
python scripts/06_export_optimal_results.py
```
> Note: Publication-ready figures and statistical CSVs will output to `results/optimal_q1_gallery/` and `results/tables/`.

## ⏭️ Next Steps
- **Phase 4**: Integrating the Dynamical Consistency Network (DCN) and Physics-Guided Latent ODE (on current CWRU benchmarks).
- **Phase 5**: Validating the architecture sequentially upon NASA IMS (Run-to-failure) and XJTU-SY datasets.
