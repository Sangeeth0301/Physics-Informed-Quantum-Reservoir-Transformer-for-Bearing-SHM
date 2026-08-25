# Phase 3: Projected Quantum Kernel Reservoir (PQKR) Evaluation vs RBF

import sys
import os

# Ensure root is in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

from src.quantum.pqkr import PQKR
from src.quantum.metrics import frobenius_divergence, compute_mmd, kernel_eigenvalue_spectrum
from src.data_prep.signal_processing import preprocess_bearing_signal
from pydmd import MrDMD, DMD

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'plots'))
os.makedirs(results_dir, exist_ok=True)

# 1. mrDMD Feature Vector Construction
def hankelize(signal, delay):
    n = len(signal)
    snapshots = np.zeros((delay, n - delay + 1))
    for i in range(delay):
        snapshots[i, :] = signal[i:i + n - delay + 1]
    return snapshots

def extract_features(signal, delay=60, svd_rank=12, max_level=3, max_cycles=6):
    """
    Constructs a fixed-length real-valued feature vector directly from mrDMD Koopman eigenvalues.
    Top-4 mode eigenvalues (by magnitude) + base Koopman metrics = dimension exactly 15.
    Eigenvalues sorted by descending magnitude before feature extraction to guarantee determinism.
    """
    signal = preprocess_bearing_signal(signal)
    hankel_H = hankelize(signal, delay)
    
    try:
        model = MrDMD(DMD(svd_rank=svd_rank), max_level=max_level, max_cycles=max_cycles)
        model.fit(hankel_H)
        eigs = model.eigs
    except Exception:
        eigs = np.array([])
        
    feats = []
    
    # Base Koopman metric extraction
    if len(eigs) > 0:
        radii = np.abs(eigs)
        spectral_radius = np.max(radii)
        unstable_ratio = np.sum(radii > 1.0) / len(radii)
        mean_freq = np.mean(np.abs(np.imag(eigs)))
    else:
        spectral_radius, unstable_ratio, mean_freq = 0.0, 0.0, 0.0
        
    feats.extend([spectral_radius, unstable_ratio, mean_freq])
    
    # Multi-resolution real representation (top-4 eigenvalues by energy)
    if len(eigs) > 0:
        # STRICT DETERMINISM: Sort primarily by magnitude (descending), 
        # then by real part (descending), then imaginary (descending) to avoid any ties.
        magnitudes = np.abs(eigs)
        reals = np.real(eigs)
        imags = np.imag(eigs)
        # lexsort sorts by the last key first. Negate to sort descending.
        sorted_idx = np.lexsort((-imags, -reals, -magnitudes))
        top_eigs = eigs[sorted_idx][:4]
    else:
        top_eigs = []
        
    # Standardize dimensions mapping complex eigenvalues purely real
    for _ in range(4 - len(top_eigs)):
        top_eigs = np.append(top_eigs, 0.0 + 0.0j)
        
    for e in top_eigs:
        feats.append(np.real(e))
        feats.append(np.imag(e))
        feats.append(np.abs(e))
        
    return np.array(feats, dtype=float)

if __name__ == "__main__":
    print("\n==================================")
    print("PHASE 3: PQKR KERNEL CONSTRUCTION")
    print("==================================\n")

    # Load previously locked preprocessed windows
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    healthy_windows = np.load(os.path.join(processed_dir, "healthy_windows.npy"))
    fault_windows   = np.load(os.path.join(processed_dir, "fault_windows.npy"))

    print("Step 1: Constructing mrDMD features... (Lightweight demo using 30 windows each)")
    num_windows = min(30, len(fault_windows), len(healthy_windows))

    X_healthy_raw = []
    X_fault_raw = []

    for i in range(num_windows):
        X_healthy_raw.append(extract_features(healthy_windows[i]))
        X_fault_raw.append(extract_features(fault_windows[i]))
        
    X_healthy_raw = np.array(X_healthy_raw)
    X_fault_raw = np.array(X_fault_raw)

    print(f"Constructed raw feature tensor shape: {X_healthy_raw.shape}")

    # 2. Z-Score Normalization (MANDATORY LEAKAGE-SAFE: Fit on Healthy ONLY)
    print("Step 2: Z-Score feature normalization...")
    scaler = StandardScaler()
    X_healthy_norm = scaler.fit_transform(X_healthy_raw)
    X_fault_norm = scaler.transform(X_fault_raw)

    # 3. PCA Compression (Required due to high dimension -> 15 > 4)
    n_qubits = 4
    print(f"Step 3: Fitting strict PCA compression down to {n_qubits} qubits...")
    pca = PCA(n_components=n_qubits, random_state=42)
    X_healthy_pca = pca.fit_transform(X_healthy_norm)
    X_fault_pca = pca.transform(X_fault_norm)

    print(f"PCA Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

    print("\n--- Running Multi-Seed Evaluation ---")

    seeds = list(range(10))
    quantum_mmd_list = []
    quantum_frob_list = []
    classical_mmd_list = []
    classical_frob_list = []

    best_seed = 0
    best_Q_H, best_Q_F, best_Q_combined = None, None, None

    # Classical Evaluation mapped identically to quantum subsets
    gamma = 1.0 / n_qubits  # Equivalent encoding scaling 
    C_H = rbf_kernel(X_healthy_pca, X_healthy_pca, gamma=gamma)
    C_F = rbf_kernel(X_fault_pca, X_fault_pca, gamma=gamma)
    C_HF = rbf_kernel(X_healthy_pca, X_fault_pca, gamma=gamma)
    
    c_frob = frobenius_divergence(C_H, C_F)
    c_mmd = compute_mmd(C_H, C_F, C_HF)

    for i in seeds:
        classical_frob_list.append(c_frob)
        classical_mmd_list.append(c_mmd)

    for seed in seeds:
        # Step 4 & 5: Circuit and Fidelity Mapping Kernel
        pqkr = PQKR(n_qubits=n_qubits, n_layers=2, seed=seed)
        
        Q_H = pqkr.compute_kernel(X_healthy_pca, X_healthy_pca)
        Q_F = pqkr.compute_kernel(X_fault_pca, X_fault_pca)
        Q_HF = pqkr.compute_kernel(X_healthy_pca, X_fault_pca)
        
        q_frob = frobenius_divergence(Q_H, Q_F)
        q_mmd = compute_mmd(Q_H, Q_F, Q_HF)
        
        quantum_frob_list.append(q_frob)
        quantum_mmd_list.append(q_mmd)
        
        if seed == 0:
            best_Q_H = Q_H
            best_Q_F = Q_F

    print(f"\nResults Verification ({len(seeds)} Seeds):")
    print(f"Quantum Frobenius:   {np.mean(quantum_frob_list):.4f} ± {np.std(quantum_frob_list):.4f}")
    print(f"Quantum MMD:         {np.mean(quantum_mmd_list):.4f} ± {np.std(quantum_mmd_list):.4f}")
    print(f"Classical Frobenius: {np.mean(classical_frob_list):.4f} ± {np.std(classical_frob_list):.4f}")
    print(f"Classical MMD:       {np.mean(classical_mmd_list):.4f} ± {np.std(classical_mmd_list):.4f}")

    # Compute diagnostic intra/inter similarities
    mean_intra_H = (np.sum(best_Q_H) - np.trace(best_Q_H)) / (num_windows*(num_windows-1))
    mean_intra_F = (np.sum(best_Q_F) - np.trace(best_Q_F)) / (num_windows*(num_windows-1))
    Q_HF_seed0 = PQKR(n_qubits=n_qubits, n_layers=2, seed=0).compute_kernel(X_healthy_pca, X_fault_pca)
    mean_inter_HF = np.mean(Q_HF_seed0)

    print(f"\nIntra-Healthy Similarity (Seed 0): {mean_intra_H:.4f}")
    print(f"Intra-Fault Similarity (Seed 0): {mean_intra_F:.4f}")
    print(f"Inter-Class Similarity (Seed 0): {mean_inter_HF:.4f}")

    print("\n--- Quick Sanity Checks (Seed 0) ---")
    is_symmetric = np.allclose(best_Q_H, best_Q_H.T, atol=1e-6)
    is_diag_one = np.allclose(np.diag(best_Q_H), 1.0, atol=1e-6)
    has_nans = np.isnan(best_Q_H).any()
    
    print(f"Kernel symmetric? {is_symmetric}")
    print(f"Kernel diagonals approx. 1? {is_diag_one}")
    print(f"Contains NaNs? {has_nans}")


    print("\n--- Running Qubit Ablation Analysis ---")
    ablation_qubits = [4, 5, 6]
    ablation_mmd = []
    
    for n_q in ablation_qubits:
        print(f"Testing {n_q} qubits...")
        # Need to re-run PCA to extract n_q components
        pca_nq = PCA(n_components=n_q, random_state=42)
        X_H_pca_nq = pca_nq.fit_transform(X_healthy_norm)
        X_F_pca_nq = pca_nq.transform(X_fault_norm)
        
        # Using seed 0 for ablation stability
        pqkr_nq = PQKR(n_qubits=n_q, n_layers=2, seed=0)
        Q_H_nq = pqkr_nq.compute_kernel(X_H_pca_nq, X_H_pca_nq)
        Q_F_nq = pqkr_nq.compute_kernel(X_F_pca_nq, X_F_pca_nq)
        Q_HF_nq = pqkr_nq.compute_kernel(X_H_pca_nq, X_F_pca_nq)
        
        mmd_nq = compute_mmd(Q_H_nq, Q_F_nq, Q_HF_nq)
        ablation_mmd.append(mmd_nq)
        print(f"  -> {n_q} qubits MMD: {mmd_nq:.4f}")

    print("\nPhase-3 Visualizations Compiling...")
    
    # Ablation Plot
    plt.figure(figsize=(6, 5))
    plt.plot(ablation_qubits, ablation_mmd, marker='o', color='purple', linewidth=2)
    plt.title("Qubit Ablation: Scalability Analysis")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Maximum Mean Discrepancy (MMD)")
    plt.xticks(ablation_qubits)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "qubit_ablation.png"), dpi=300)
    plt.close()

    # Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(best_Q_H, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title("Healthy Quantum Kernel Heatmap")
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(best_Q_F, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title("Fault Quantum Kernel Heatmap")
    fig.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "quantum_kernel_heatmaps.png"), dpi=300)
    plt.close()

    # Kernel Eigenvalue Spectrum
    eig_Q_H = kernel_eigenvalue_spectrum(best_Q_H)
    eig_Q_F = kernel_eigenvalue_spectrum(best_Q_F)

    plt.figure(figsize=(8, 5))
    plt.plot(eig_Q_H, marker='o', label="Healthy Kernel", color='green', alpha=0.8)
    plt.plot(eig_Q_F, marker='x', label="Fault Kernel", color='red', alpha=0.8)
    plt.title("Quantum Kernel Eigenvalue Spectrum")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "kernel_eigenvalue_spectrum.png"), dpi=300)
    plt.close()

    # Divergence Comparison Bar Plot
    labels = ['Quantum (PQKR)', 'Classical (RBF)']
    means_mmd = [np.mean(quantum_mmd_list), np.mean(classical_mmd_list)]
    std_mmd = [np.std(quantum_mmd_list), np.std(classical_mmd_list)]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, means_mmd, yerr=std_mmd, capsize=10, color=['purple', 'gray'], alpha=0.8)
    plt.title("MMD Separation Comparison")
    plt.ylabel("Maximum Mean Discrepancy (MMD)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mmd_divergence_comparison.png"), dpi=300)
    plt.close()

    print("\n=============================================")
    print("IEEE FORMATTED TABULAR RESULTS GENERATED")
    print("=============================================\n")
    
    tables_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'tables'))
    os.makedirs(tables_dir, exist_ok=True)

    # 1. Main Divergence Table
    summary_df = pd.DataFrame({
        "Method": ["Classical RBF Kernel", "Projected Quantum Kernel (PQKR)"],
        "Mean Frobenius Divergence": [
            f"{np.mean(classical_frob_list):.4f} ± {np.std(classical_frob_list):.4f}",
            f"{np.mean(quantum_frob_list):.4f} ± {np.std(quantum_frob_list):.4f}"
        ],
        "Maximum Mean Discrepancy (MMD)": [
            f"{np.mean(classical_mmd_list):.4f} ± {np.std(classical_mmd_list):.4f}",
            f"{np.mean(quantum_mmd_list):.4f} ± {np.std(quantum_mmd_list):.4f}"
        ]
    })
    
    # 2. Ablation Table
    ablation_df = pd.DataFrame({
        "Qubits (PCA Components)": ablation_qubits,
        "Measured Quantum MMD": [f"{mmd:.4f}" for mmd in ablation_mmd]
    })
    
    # Print formatted markdown to console
    print(summary_df.to_markdown(index=False))
    print("\n")
    print(ablation_df.to_markdown(index=False))
    
    # Export for Q1 Publication
    summary_df.to_csv(os.path.join(tables_dir, "divergence_metrics.csv"), index=False)
    ablation_df.to_csv(os.path.join(tables_dir, "qubit_ablation.csv"), index=False)
    
    # Generate explicit IEEE-style LaTeX code using Styler formatting to prevent future deprecation warnings
    with open(os.path.join(tables_dir, "divergence_metrics.tex"), "w") as f:
        f.write(summary_df.style.to_latex(
            caption="Comparative Divergence Metrics over 10 seeded iterations",
            label="tab:divergence_metrics",
            hrules=True
        ))
        
    with open(os.path.join(tables_dir, "qubit_ablation.tex"), "w") as f:
        f.write(ablation_df.style.to_latex(
            caption="Quantum Kernel Separation scalability across Qubit Dimensionality",
            label="tab:qubit_ablation",
            hrules=True
        ))
        
    print("\n[+] Tables exported to results/tables/ in CSV and LaTeX format for direct manuscript inclusion.")

    print("\nPQKR PHASE COMPLETE — READY FOR REVIEW")
