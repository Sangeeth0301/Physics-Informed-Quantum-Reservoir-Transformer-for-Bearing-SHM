# Phase 4: Detailed Architecture & Execution Specification
**Quantum-Enhanced Koopman Operator Learning for Incipient Bearing Instability**

This document serves as the absolute, comprehensive engineering specification for Phase 4. It details the exact mathematical corrections, the deep architectural design of the Dynamical Consistency Network (DCN) and Physics-Guided Latent ODE, and the precise step-by-step pipeline we will execute across both the CWRU and NASA IMS datasets.

---

## 1. The Core Architectural Refinements (Solving the "Lags")

To ensure this framework achieves Q1 publication standards (e.g., IEEE T-IE, MSSP), we must address three specific mathematical disconnects in standard hybrid architectures.

### 1.1 The DCN Topology: Dense Autoencoder vs. 1D-CNN
*   **The Problem:** The standard approach suggests feeding raw time-series data into a 1D-CNN. However, our Phase 3 Projected Quantum Kernel Reservoir (PQKR) does not output a time-series. It outputs highly abstract, unit-normalized **Quantum State Vectors** (e.g., a 32-dimensional complex amplitude vector from the Hilbert space) or dense Fidelity block matrices. A Convolutional Neural Network (CNN) relies on translation invariance (finding a pattern anywhere in time), which is meaningless when applied to a fixed-dimensional quantum state vector representing a global geometric point.
*   **The Solution:** The DCN must be implemented as a **Dense (Fully Connected) Autoencoder**.
*   **Mathematical Flow:** 
    *   **Input ($X_q$):** The concatenated real/imaginary components of the PQKR state vector $X_q \in \mathbb{R}^{2^N}$ (where $N$ is the number of qubits).
    *   **Encoder:** $Z = \text{ELU}(W_2(\text{ELU}(W_1 X_q + b_1)) + b_2)$, compressing the quantum state into a low-dimensional bottleneck ($Z \in \mathbb{R}^k$).
    *   **Decoder:** $\hat{X}_q = \text{Dense}(Z)$, attempting to reconstruct the quantum geometry.
    *   **Loss:** $\mathcal{L}_{Recon} = ||X_q - \hat{X}_q||^2_2$. 
*   **Why it works:** By training *strictly* on healthy data, the Dense DCN learns the exact high-dimensional topological manifold of a "stable" bearing. When an incipient fault alters the quantum state geometry, the DCN's reconstruction error ($\mathcal{L}_{Recon}$) will deterministically spike.

### 1.2 Physics-Guided Neural ODE: The Dimensionality Projection
*   **The Problem:** We intend to constrain the latent space using continuous bearing physics (the Jeffcott Rotor with nonlinear Hertzian contact). A standard Neural ODE evolves an abstract vector $\frac{dz}{dt} = f_\theta(z, t)$. However, physical ODEs demand strictly physical variables: displacement ($x$) and velocity ($\dot{x}$). We cannot calculate a physical residual force $F = m\ddot{x} + c\dot{x} + kx + k_{hertz}x^{3/2}$ on an arbitrary 64-dimensional latent tensor.
*   **The Solution:** We will enforce a **Physical Coordinate Projection** inside the DCN bottleneck.
*   **Mathematical Flow:**
    *   Let the DCN bottleneck be $Z = [z_1, z_2, z_3, ..., z_k]$.
    *   We explicitly map: $x_{pseudo} = z_1$ and $\dot{x}_{pseudo} = z_2$.
    *   The Neural ODE (via `torchdiffeq`) evolves $[z_1, z_2]$ forward in time.
    *   The **Physics Residual Loss ($\mathcal{L}_{Physics}$)** is computed *only* on the $[z_1, z_2]$ trajectory using the driven oscillator equation: $\ddot{x} + 2\zeta\omega_n\dot{x} + \omega_n^2x + \alpha x^{3/2} = 0$.
    *   The remaining latent dimensions $[z_3...z_k]$ are evolved via a standard unconstrained neural network to absorb unknown friction, noise, and complex environmental dynamics that the Jeffcott model cannot capture.

### 1.3 The Fusion Trap: Calculating the Instability Score (SI)
*   **The Problem:** The final "Change-Point" detection (the SI Score) requires fusing multiple independent metrics: DCN Error, Koopman Drift, ODE Physics Residual, and Quantum MMD. These metrics exist on vastly different scales. (e.g., MMD $\approx 0.5$, Koopman Frequencies $\approx 2500$, DCN Error $\approx 0.0001$). A simple weighted sum guarantees that the largest absolute number will entirely drown out the sensitivity of the quantum and physics layers.
*   **The Solution:** A **Running Mahalanobis Distance (Z-Score) Fusion Layer**.
*   **Mathematical Flow:**
    1. During healthy training, we calculate the mean ($\mu$) and standard deviation ($\sigma$) for every individual metric.
    2. During testing, every new metric $m_i$ is standardized: $Z_i = \frac{m_i - \mu_i}{\sigma_i}$.
    3. The final Instability Score is a bounded fusion (e.g., via Softmax or Sigmoid): $SI = \sigma(\sum_{i} w_i Z_i)$.
*   **Why it works:** Every deviation is now measured in "Standard Deviations away from Healthy." A $3\sigma$ spike in the tiny DCN error carries the exact same weight as a $3\sigma$ spike in the massive Koopman frequency, ensuring our quantum sensitivity is perfectly preserved in the final alarm.

---

## 2. Immediate Execution Pipeline (The Next Steps)

We must prove this composite architecture on the **CWRU dataset** before attempting the massive temporal logic of the IMS dataset.

### Step A: Implement and Validate on CWRU (Current Task)
1. **Intake Phase 4 Code:** You (the developer) provide the baseline `04_dcn_physics_ode.py`.
2. **Architectural Retrofit:** I will adapt the code to include the Dense DCN, the 2D Physical ODE projection, and the Z-Score normalization logic described above.
3. **Training Phase:** We will train the DCN and Neural ODE *exclusively* on the healthy CWRU window features extracted during Phase 1-3.
4. **Testing Phase:** We will pass both the Healthy and 7-mil Fault datasets through the frozen architecture.
5. **Output Generation:** We will generate Q1-ready graphics:
    *   `q1_dcn_reconstruction.png`: Showing the MSE divergence between classes.
    *   `q1_latent_physics_ode.png`: A phase-portrait of the $[z_1, z_2]$ ODE evolution vs. the true constraint.
    *   `q1_instability_score.png`: The final SI curve demonstrating clear separability.

### Step B: The Transition to NASA IMS (Run-To-Failure)
Once Step A is validated, we move to the definitive proof: The NASA IMS dataset. 

*   **The Challenge:** IMS contains a 1-second vibration file recorded every 10 minutes continuously for 35 days until catastrophic failure. Standard overlapped windowing (like we used for CWRU) would generate hundreds of millions of tensors, causing an immediate Out-Of-Memory (OOM) system crash.
*   **The Strategy (Temporal Striding):** We will build `notebooks/01_load_ims_and_plot.py`. We will extract exactly *one continuous 2048-sample window* per 10-minute file. 
*   **The Pipeline:** We will run this sequential, thinned dataset through the entire Phase 1 $\rightarrow$ Phase 4 pipeline.
*   **The Ultimate Result:** We will plot the SI Score against the 35-day timeline. It will prove to reviewers mathematically that our Quantum-Physics framework flags an incipient $\sigma$-deviation **days or weeks before** traditional FFT methods show physical fault frequencies.

---
*Ready to begin execution of Step A.*
