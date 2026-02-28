# Full Project Explanation

### 1. Project Title (Final & Publication-Ready)

**Quantum-Enhanced Koopman Operator Learning via Projected Quantum Kernel Reservoir and Physics-Guided Latent Hamiltonian Dynamics for Incipient Bearing Instability Early Warning**

### 2. General Idea of the Project

We are building an **advanced hybrid quantum-classical system** that can detect **very early (incipient) faults** in rolling element bearings of rotating machines (motors, turbines, pumps, etc.) **much earlier** than traditional methods.

- **Input**: Vibration signals from accelerometers (horizontal + vertical).
- **Core Innovation**: Instead of waiting for obvious fault frequencies (BPFO, BPFI, BSF) to appear, we detect the **birth of dynamical instability** — subtle changes in the underlying dynamics of the bearing (mode splitting, spectral radius drift) while the signal still looks almost normal.
- **Approach**: Train only on healthy data (unsupervised). Use physics rules of bearing motion, multi-scale dynamics analysis, and quantum computing to make tiny instability signals stand out.
- **Output**: A continuous "Instability Score" (SI) that gives an early warning before the fault becomes visible in the spectrum.

This is inspired by the Yaghoubi 2024 paper (which added a simple quantum layer to a CNN for damage detection on ultrasonic images), but we have upgraded it dramatically for vibration data, real rotating machinery, and unsupervised early warning.

### 3. The Problem & The Depth Problem

**Surface Problem**  
Bearings fail silently. Incipient faults (tiny pits, micro-cracks) produce very weak periodic impulses that are buried in noise and masked by variable operating conditions (changing speed and load). By the time fault frequencies appear in the spectrum, significant damage has already happened, leading to costly downtime.

**Deep / Core Problem**  
Traditional methods (FFT, classical ML, even deep learning) fail early detection because:
- They need lots of faulty data for training (rare in real life).
- They look for frequency peaks after the fault has already grown.
- They ignore the **physics** of the bearing (nonlinear Hertzian contact, impulsive forcing).
- They lack sensitivity to **subtle dynamical instability** that starts long before the frequency peak.

This project solves the **deep problem**: Detect the **loss of dynamical stability** (the birth of instability) using only healthy data, physics constraints, and quantum enhancement.

### 4. What We Have Done So Far

**Completed**:
- Full GitHub repo setup with professional structure (src, notebooks, data/raw, data/processed, results/plots, configs).
- Venv + requirements.txt (PyTorch, PennyLane, PyDMD, SciPy, etc.).
- Downloaded and organized **CWRU dataset** (8 key .mat files: healthy 97–100, 7 mil faults 105–108).
- Notebook `01_load_cwru_and_plot.py`: Robust loading (scipy + h5py), signal extraction, full + zoomed plots showing clear periodic impulses in fault.
- Preprocessing: Bandpass filter, Hilbert envelope, mean removal, windowing (2048 samples, 512 stride), saved .npy files in `data/processed/`.
- Notebook `02_mrdmd_analysis.py`: Hankelization, standard DMD sanity check, mrDMD (multi-resolution), eigenvalues, modes, multi-window stats (spectral radius, unstable ratio, mean frequency), healthy vs fault comparison, envelope integration.
- Notebook `03_pqkr_embedding.py` (and hardened Phase 3 scripts): PQKR quantum embedding on mrDMD features, fidelity kernel, classical RBF baseline, divergence metrics, heatmaps, ablation, extensive multi-seed statistical robustness, eigenvalue KS-tests, and UMAP clustering.
- All optimal publication-grade plots saved in `results/optimal_q1_gallery/` and committed to GitHub.

**Current Status**: Phase 1, 2, and 3 are complete. We have a working pipeline up to quantum kernel divergence, fully statistically hardened and ready for manuscript insertion on CWRU data.

**Not Done Yet**: IMS and XJTU-SY datasets, Phase 4 (DCN + Physics ODE), Phase 5 (SI score), full experiments/ablation on multiple datasets, hardware runs, paper writing.

### 5. Optimal Self-Explainable Framework Diagram

```
RAW VIBRATION SIGNALS
(2-channel, variable speed/load, CWRU → IMS → XJTU-SY)

          ↓
Signal Conditioning
(Bandpass, Hilbert Envelope, Normalization, Windowing 2048/512)

          ↓
Multi-Resolution DMD (mrDMD)
(Hankelization → Modes, Frequencies, Decay Rates at multiple scales)

          ↓
Koopman Operator Estimation
(Spectral radius drift, mode splitting, attractor distortion)

          ↓
Projected Quantum Kernel Reservoir (PQKR)
(Angle encoding → Fixed random entangled circuit → Fidelity Kernel)

          ↓
Dynamical Consistency Network (DCN)
(Lightweight 1D-CNN Autoencoder → Reconstruction Deviation)

          ↓
Physics-Guided Latent Evolution
(Neural ODE constrained by nonlinear Jeffcott + Hertzian model)

          ↓
Change-Point Detection & Instability Score (SI)
(Fusion of Recon Error + Koopman Drift + Physics Residual + Quantum Divergence + mrDMD Novelty)

          ↓
EARLY INSTABILITY WARNING
```

### 6. Complete Pipeline & Usage of All Parts

1. **Signal Conditioning**: Clean raw signal, highlight impulses with envelope, segment into windows.  
   Use: Prepare clean data for dynamics.

2. **mrDMD**: Extract multi-scale modes and eigenvalues.  
   Use: Isolate weak early impulses from noise.

3. **Koopman**: Linearize dynamics and track stability drift.  
   Use: Detect instability birth mathematically.

4. **PQKR**: Project features into quantum Hilbert space and compute kernel.  
   Use: Exponential sensitivity to micro-changes (quantum novelty).

5. **DCN**: Autoencoder learns healthy reconstruction.  
   Use: Deviation = anomaly.

6. **Physics ODE**: Evolve latent state while enforcing bearing physics equation.  
   Use: Adds trust and reduces false positives.

7. **SI Score**: Weighted sum of all deviations.  
   Use: Single actionable early warning signal.

### 7. All About the Datasets

**CWRU (Current)**: Benchmark, small, variable load, 7 mil faults as incipient proxy.  
Use: Fast testing, baseline results.

**IMS (NASA Run-to-Failure)**: True progressive degradation (hours until failure).  
Use: Show real early progression over time.

**XJTU-SY**: Variable speed/load, horizontal/vertical channels.  
Use: Prove robustness to real non-stationary conditions.

**Why only CWRU so far?**  
It is the **optimal starting dataset** for rapid development and testing. It allows quick validation of the full pipeline (loading, mrDMD, PQKR) without large file handling issues. Once the core works on CWRU, we add IMS (for progression) and XJTU-SY (for variable conditions) — this is standard in Q1 papers.

**Next**: After Phase 4 (DCN + Physics ODE) works on CWRU, we add IMS.

### 8. Next Immediate Step (Phase 4)

**Phase 4: Dynamical Consistency Network (DCN) + Physics-Guided Latent ODE**

We are now entering the **physics + consistency** core — very strong for Q1.
