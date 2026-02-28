import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, ks_2samp, entropy
from sklearn.metrics.pairwise import rbf_kernel

# Set Matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.titleweight': 'bold'
})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_prep.signal_processing import preprocess_bearing_signal
from pydmd import MrDMD, DMD
from src.quantum.pqkr import PQKR
from src.quantum.metrics import compute_mmd, frobenius_divergence

# Paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
processed_dir = os.path.join(base_dir, 'data', 'processed')
results_plots = os.path.join(base_dir, 'results', 'plots')
results_tables = os.path.join(base_dir, 'results', 'tables')
os.makedirs(results_plots, exist_ok=True)
os.makedirs(results_tables, exist_ok=True)

# Feature extraction logic identical to previous phases
def hankelize(signal, delay):
    n = len(signal)
    snapshots = np.zeros((delay, n - delay + 1))
    for i in range(delay):
        snapshots[i, :] = signal[i:i + n - delay + 1]
    return snapshots

def extract_features(signal, delay=60, svd_rank=12, max_level=3, max_cycles=6):
    signal = preprocess_bearing_signal(signal)
    hankel_H = hankelize(signal, delay)
    try:
        model = MrDMD(DMD(svd_rank=svd_rank), max_level=max_level, max_cycles=max_cycles)
        model.fit(hankel_H)
        eigs = model.eigs
    except:
        eigs = np.array([])
    feats = []
    if len(eigs) > 0:
        radii = np.abs(eigs)
        spectral_radius = np.max(radii)
        unstable_ratio = np.sum(radii > 1.0) / len(radii)
        mean_freq = np.mean(np.abs(np.imag(eigs)))
    else:
        spectral_radius, unstable_ratio, mean_freq = 0.0, 0.0, 0.0
    feats.extend([spectral_radius, unstable_ratio, mean_freq])
    
    if len(eigs) > 0:
        magnitudes = np.abs(eigs)
        reals = np.real(eigs)
        imags = np.imag(eigs)
        sorted_idx = np.lexsort((-imags, -reals, -magnitudes))
        top_eigs = eigs[sorted_idx][:4]
    else:
        top_eigs = []
        
    for _ in range(4 - len(top_eigs)):
        top_eigs = np.append(top_eigs, 0.0 + 0.0j)
    for e in top_eigs:
        feats.extend([np.real(e), np.imag(e), np.abs(e)])
    return np.array(feats, dtype=float)

# Load data
healthy_windows = np.load(os.path.join(processed_dir, "healthy_windows.npy"))
fault_windows   = np.load(os.path.join(processed_dir, "fault_windows.npy"))

num_windows = min(20, len(fault_windows), len(healthy_windows)) # Keep 20 for efficient CI runs
features_H = np.array([extract_features(healthy_windows[i]) for i in range(num_windows)])
features_F = np.array([extract_features(fault_windows[i]) for i in range(num_windows)])

# Z-score normalization
scaler = StandardScaler()
X_H_norm = scaler.fit_transform(features_H)
X_F_norm = scaler.transform(features_F)

# PCA Compression
n_qubits = 5
pca = PCA(n_components=n_qubits, random_state=42)
X_H_pca = pca.fit_transform(X_H_norm)
X_F_pca = pca.transform(X_F_norm)
labels = np.array([0]*num_windows + [1]*num_windows)

# Helper Metrics
def compute_separation_ratio(intra_H, intra_F, inter_HF):
    return (intra_H + intra_F) / (2.0 * inter_HF + 1e-10)

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

print("[1] Evaluating Quantum vs Classical Seed Sensitivities & Kernel Bounds...")
seeds = list(range(10))

# Storage
q_frob, q_mmd_list = [], []
q_intraH, q_intraF, q_inter = [], [], []
q_sep_ratio = []
q_spectra_H, q_spectra_F = [], []

c_frob, c_mmd_list = [], []
c_intraH, c_intraF, c_inter = [], [], []
c_sep_ratio = []
c_spectra_H, c_spectra_F = [], []

seed_metrics = []

for s in seeds:
    pqkr = PQKR(n_qubits=n_qubits, n_layers=2, seed=s)
    
    # Quantum
    K_HH = pqkr.compute_kernel(X_H_pca, X_H_pca)
    K_FF = pqkr.compute_kernel(X_F_pca, X_F_pca)
    K_HF = pqkr.compute_kernel(X_H_pca, X_F_pca)
    
    q_frob.append(frobenius_divergence(K_HH, K_FF))
    q_mmd_list.append(compute_mmd(K_HH, K_FF, K_HF))
    
    iH = (np.sum(K_HH) - np.trace(K_HH)) / (num_windows*(num_windows-1))
    iF = (np.sum(K_FF) - np.trace(K_FF)) / (num_windows*(num_windows-1))
    iHF = np.sum(K_HF) / (num_windows*num_windows)
    
    q_intraH.append(iH)
    q_intraF.append(iF)
    q_inter.append(iHF)
    sr = compute_separation_ratio(iH, iF, iHF)
    q_sep_ratio.append(sr)
    
    eigs_qH = np.real(np.linalg.eigvals(K_HH))
    eigs_qH.sort()
    q_spectra_H.append(eigs_qH[::-1])
    eigs_qF = np.real(np.linalg.eigvals(K_FF))
    eigs_qF.sort()
    q_spectra_F.append(eigs_qF[::-1])
    
    seed_metrics.append({
        "seed": s, "method": "Quantum", "frobenius": q_frob[-1], "mmd": q_mmd_list[-1],
        "intra_similarity": (iH+iF)/2, "inter_similarity": iHF, "separation_ratio": sr
    })
    
    # Classical
    gamma_val = 1.0 / n_qubits
    C_HH = rbf_kernel(X_H_pca, X_H_pca, gamma=gamma_val)
    C_FF = rbf_kernel(X_F_pca, X_F_pca, gamma=gamma_val)
    C_HF = rbf_kernel(X_H_pca, X_F_pca, gamma=gamma_val)
    
    c_frob.append(frobenius_divergence(C_HH, C_FF))
    c_mmd_list.append(compute_mmd(C_HH, C_FF, C_HF))
    
    ciH = (np.sum(C_HH) - np.trace(C_HH)) / (num_windows*(num_windows-1))
    ciF = (np.sum(C_FF) - np.trace(C_FF)) / (num_windows*(num_windows-1))
    ciHF = np.sum(C_HF) / (num_windows*num_windows)
    csr = compute_separation_ratio(ciH, ciF, ciHF)
    
    c_intraH.append(ciH)
    c_intraF.append(ciF)
    c_inter.append(ciHF)
    c_sep_ratio.append(csr)
    
    seed_metrics.append({
        "seed": s, "method": "Classical", "frobenius": c_frob[-1], "mmd": c_mmd_list[-1],
        "intra_similarity": (ciH+ciF)/2, "inter_similarity": ciHF, "separation_ratio": csr
    })

pd.DataFrame(seed_metrics).to_csv(os.path.join(results_tables, "per_seed_metrics_phase3.csv"), index=False)

# Seed Sensitivity Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(seeds, q_mmd_list, marker='o', label='Quantum PQKR MMD', color='#2ca02c')
ax.plot(seeds, c_mmd_list, marker='s', label='Classical RBF MMD', color='#d62728', linestyle='--')
ax.set_title("(a) Seed Sensitivity & Separability Bound Analysis")
ax.set_xlabel("Random Seed $s$")
ax.set_ylabel("Maximum Mean Discrepancy (MMD)")
ax.set_ylim(0, max(max(q_mmd_list), max(c_mmd_list)) + 0.1)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_plots, "q1_seed_sensitivity.png"))
plt.close()

print("[2] Noise Robustness & Window Subsampling...")
# Noise injection test (Gaussian noise on PCA components)
noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
q_noise_mmd, c_noise_mmd = [], []

for noise in noise_levels:
    q_m, c_m = [], []
    for s in range(5): # 5 seeds per noise level
        X_H_noisy = X_H_pca + np.random.normal(0, noise, X_H_pca.shape)
        X_F_noisy = X_F_pca + np.random.normal(0, noise, X_F_pca.shape)
        
        pq = PQKR(n_qubits=n_qubits, n_layers=2, seed=s)
        K_HH = pq.compute_kernel(X_H_noisy, X_H_noisy)
        K_FF = pq.compute_kernel(X_F_noisy, X_F_noisy)
        K_HF = pq.compute_kernel(X_H_noisy, X_F_noisy)
        q_m.append(compute_mmd(K_HH, K_FF, K_HF))
        
        C_HH = rbf_kernel(X_H_noisy, X_H_noisy, gamma=1.0/n_qubits)
        C_FF = rbf_kernel(X_F_noisy, X_F_noisy, gamma=1.0/n_qubits)
        C_HF = rbf_kernel(X_H_noisy, X_F_noisy, gamma=1.0/n_qubits)
        c_m.append(compute_mmd(C_HH, C_FF, C_HF))
        
    q_noise_mmd.append(np.mean(q_m))
    c_noise_mmd.append(np.mean(c_m))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(noise_levels, q_noise_mmd, marker='o', label='Quantum PQKR', color='blue')
ax.plot(noise_levels, c_noise_mmd, marker='s', label='Classical RBF', color='orange', linestyle='--')
ax.set_title("(b) Gaussian Noise Robustness Test")
ax.set_xlabel("Noise Variance $\sigma^2$")
ax.set_ylabel("MMD Separability")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_plots, "q1_noise_robustness.png"))
plt.close()

# Spectral Entropy & Eigenspectrum
mean_spec_H = np.mean(np.array(q_spectra_H), axis=0)
mean_spec_F = np.mean(np.array(q_spectra_F), axis=0)
std_spec_H = np.std(np.array(q_spectra_H), axis=0)
std_spec_F = np.std(np.array(q_spectra_F), axis=0)

# Shift and normalize for entropy
norm_H = np.abs(mean_spec_H) / np.sum(np.abs(mean_spec_H))
norm_F = np.abs(mean_spec_F) / np.sum(np.abs(mean_spec_F))
ent_H = entropy(norm_H + 1e-12)
ent_F = entropy(norm_F + 1e-12)

cum_energy_H = np.cumsum(mean_spec_H) / np.sum(mean_spec_H)
cum_energy_F = np.cumsum(mean_spec_F) / np.sum(mean_spec_F)
_, ks_p = ks_2samp(mean_spec_H, mean_spec_F)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(np.log10(mean_spec_H + 1e-12), label=f'Healthy', color='green', linewidth=2)
ax1.fill_between(range(len(mean_spec_H)), np.log10(mean_spec_H - std_spec_H + 1e-12), np.log10(mean_spec_H + std_spec_H + 1e-12), color='green', alpha=0.2)
ax1.plot(np.log10(mean_spec_F + 1e-12), label=f'Fault', color='red', linewidth=2)
ax1.fill_between(range(len(mean_spec_F)), np.log10(mean_spec_F - std_spec_F + 1e-12), np.log10(mean_spec_F + std_spec_F + 1e-12), color='red', alpha=0.2)
ax1.set_title(f"(c) Quantum Kernel Eigenspectrum Decay\nKS-Test $p={ks_p:.2e}$")
ax1.set_xlabel("Eigenvalue Index $i$")
ax1.set_ylabel("$\log_{10}(\lambda_i)$")
ax1.legend()

ax2.plot(cum_energy_H, label=f'Healthy (Ent={ent_H:.2f})', color='green', linewidth=2)
ax2.plot(cum_energy_F, label=f'Fault (Ent={ent_F:.2f})', color='red', linewidth=2)
ax2.axhline(0.95, color='black', linestyle=':', label='95% Energy Bound')
ax2.set_title(f"(d) Cumulative Spectral Energy Dynamics")
ax2.set_xlabel("Eigenvalue Index $i$")
ax2.set_ylabel("Cumulative Explained Variance")
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_plots, "q1_spectral_validation.png"))
plt.close()

# Compile Advantages into Summary
print("[3] Compiling Overall Q1 Advantage Metrics...")
cohens_mmd = cohens_d(q_mmd_list, c_mmd_list)
cohens_frob = cohens_d(q_frob, c_frob)

summary_data = {
    "Metric": ["Frobenius Divergence", "MMD", "Intra Similar", "Inter Similar", "Separation Ratio"],
    "Quantum_Mean": [np.mean(q_frob), np.mean(q_mmd_list), np.mean(q_intraH+q_intraF)/2, np.mean(q_inter), np.mean(q_sep_ratio)],
    "Quantum_Std": [np.std(q_frob), np.std(q_mmd_list), np.std(q_intraH+q_intraF)/2, np.std(q_inter), np.std(q_sep_ratio)],
    "Classical_Mean": [np.mean(c_frob), np.mean(c_mmd_list), np.mean(c_intraH+c_intraF)/2, np.mean(c_inter), np.mean(c_sep_ratio)],
    "Classical_Std": [np.std(c_frob), np.std(c_mmd_list), np.std(c_intraH+c_intraF)/2, np.std(c_inter), np.std(c_sep_ratio)],
    "Effect_Size_D": [cohens_frob, cohens_mmd, np.nan, np.nan, np.nan]
}
pd.DataFrame(summary_data).to_csv(os.path.join(results_tables, "aggregated_summary_phase3.csv"), index=False)

# Annotated Heatmaps update
pqkr0 = PQKR(n_qubits=n_qubits, n_layers=2, seed=0)
K_HH0 = pqkr0.compute_kernel(X_H_pca, X_H_pca)
K_FF0 = pqkr0.compute_kernel(X_F_pca, X_F_pca)
K_HF0 = pqkr0.compute_kernel(X_H_pca, X_F_pca)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
im1 = axs[0].imshow(K_HH0, cmap='viridis', aspect='auto')
axs[0].set_title(f"(e) Healthy Intra-Class\n$\\mu_{{intra}}={np.mean(q_intraH):.3f}$")
plt.colorbar(im1, ax=axs[0], label=r"Fidelity $|\langle\psi_i | \psi_j\rangle|^2$")

im2 = axs[1].imshow(K_FF0, cmap='viridis', aspect='auto')
axs[1].set_title(f"(f) Fault Intra-Class\n$\\mu_{{intra}}={np.mean(q_intraF):.3f}$")
plt.colorbar(im2, ax=axs[1], label=r"Fidelity $|\langle\psi_i | \psi_j\rangle|^2$")

im3 = axs[2].imshow(K_HF0, cmap='magma', aspect='auto')
axs[2].set_title(f"(g) Cross-Class Density\n$\\mu_{{inter}}={np.mean(q_inter):.3f}$, SepRatio={np.mean(q_sep_ratio):.2f}")
plt.colorbar(im3, ax=axs[2], label=r"Fidelity $|\langle\psi_x | \psi_y\rangle|^2$")

plt.tight_layout()
plt.savefig(os.path.join(results_plots, "q1_final_quantum_kernel_matrices.png"))
plt.close()

print("PHASE 3 HARDENING COMPLETE \u2014 READY FOR PHASE 4")
