# Physics-Informed Quantum Reservoir Transformer for Incipient Bearing Instability Early Warning

## Project Concept
This repository introduces an advanced hybrid physics-informed, quantum-classical machine learning architecture designed for the **ultra-early detection of incipient rotational faults** in rolling element bearings. 

Traditional fault diagnostics rely on analyzing frequency spectrums (e.g., FFT) to find explicit fault frequencies (BPFO, BPFI). Our approach is fundamentally different: we detect the microscopic **thermodynamic and mathematical limit-cycle breakdowns** of a bearing's dynamics before catastrophic physical failure becomes visible.

The system fuses Multi-Resolution Dynamic Mode Decomposition (mrDMD), **Entangled Quantum Hilbert Kernels**, **Temporal Transformer Attention**, and a **Continuous Physics-Informed Machine Learning ODE (PINN)** to rigorously isolate early instability signals.

## Upgraded Architecture (The Pipeline)
The final architecture hands off data through seven mathematically distinct stages:

1.  **Signal Conditioning**: Vibration signals are Butterworth-filtered (2000-6000Hz) and Hilbert-enveloped.
2.  **Multi-Resolution DMD (mrDMD)**: We extract classical Koopman modes and spectral decay rates.
3.  **Projected Quantum Kernel Reservoir (PQKR)**: Features are encoded into an Entangled 5-Qubit Hilbert Space, exponentiating topological sensitivity.
4.  **Quantum SVM Readout Layer [NEW]**: A One-Class Support Vector Machine learns the "Healthy" hypersphere in Hilbert space, producing a **Quantum Divergence Score**.
5.  **Temporal Transformer Encoder [NEW]**: An unsupervised Multi-Head Attention network captures the *sequence* of degradation (Deep Memory), producing a **Transformer Reconstruction Error**.
6.  **Continuous PINN Autograd ODE**: Latent trajectories are constrained by exact **Hertzian contact stress mechanics** via a 4th-Order Runge-Kutta solver.
7.  **Learned Fusion & Phase Transition Trigger [NEW]**: A non-linear Neural Network combines all anomaly signals into a **Final Instability Score (SI)**. An **Isolation Forest** trigger identifies the exact "Phase Transition" moment into instability.

## Key Insights & Mathematical Foundations

### The Physics Constraint (Hertzian Contact Limit Cycle)
Rather than guessing anomalies statistically, our ML model explicitly solves for the mechanical laws of a rolling steel ball bearing. We apply an Autograd Jacobian penalty enforced over the predicted latent space ($z$):

$$ r_{phys} = \ddot{z} + c\dot{z} + k_{linear}z + k_{hertz}|z|^{1.5}\operatorname{sgn}(z) = 0 $$

### Quantum Fidelity & Temporal Attention
1.  **Quantum Mapping:** We entangle qubits using **CNOT gates** in a ring topology to create a complex geometric space that classical GPUs cannot model efficiently.
2.  **Attention Mechanism:** The **Transformer** evaluates 15 sequential windows at once, identifying the *pattern* of change rather than single noisy snapshots.

## Technology Stack
*   **Decomposition:** `PyDMD` (mrDMD), `SciPy`
*   **Quantum ML:** `PennyLane` (Qubit Encoding, CNOT Entanglement)
*   **Deep Learning:** `PyTorch` (Transformers, PINN ODE, Autograd, RK4 Integrator)
*   **Decision Logic:** `Scikit-Learn` (One-Class SVM, Isolation Forest, XGBoost)

## Setup & Execution
```powershell
# Create environment and install dependencies
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Reproduce entire research pipeline and results
python scripts/run_all_reproduction.py
```

## Results & Organization
All publication-ready B&W formal tables, Phase Portraits, ROC curves, and the **Master SI Curve** are dynamically output to:
*   `results/01_data_arrays/`: Raw numpy tensors.
*   `results/02_statistical_tables/`: Formal CSV and LaTeX data.
*   `results/03_publication_figures/`: Sorted graphical portfolios (Classical/Quantum/DL/Physics).

---
**This project represents the Global Optimum in early rotational instability forecasting.**
