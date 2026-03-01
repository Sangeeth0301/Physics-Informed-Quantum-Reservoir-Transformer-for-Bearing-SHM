import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import ttest_ind, ks_2samp
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
    'font.serif': ['Times New Roman']
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

# Feature extraction logic from Phase 3
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

num_windows = min(20, len(fault_windows), len(healthy_windows)) # Using 20 for efficient run
np.random.seed(42) # Determinism for classical parts

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
X_combined_pca = np.vstack((X_H_pca, X_F_pca))


def compute_kernel_matrix(X1, X2, kernel_func):
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_func(X1[i], X2[j])
    return K

# Helper: Separation Ratio
def compute_separation_ratio(intra_H, intra_F, inter_HF):
    return (intra_H + intra_F) / (2.0 * inter_HF + 1e-10)

# HELPER: CI
def calc_ci(data):
    return 1.96 * np.std(data) / np.sqrt(len(data))

# TASK 1: Multi-Seed PQKR Evaluation
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Storage
q_frob_list, q_mmd_list = [], []
q_intra_H_list, q_intra_F_list, q_inter_list = [], [], []
q_sep_ratio_list = []

c_frob_list, c_mmd_list = [], []
c_intra_H_list, c_intra_F_list, c_inter_list = [], [], []
c_sep_ratio_list = []

q_sil_list, q_db_list, q_ch_list = [], [], []
c_sil_list, c_db_list, c_ch_list = [], [], []

q_spectra_H, q_spectra_F = [], []
c_spectra_H, c_spectra_F = [], []

seed_metrics = []

for s in seeds:
    # --- Add Gaussian Variance ---
    np.random.seed(s)
    X_H_noisy = X_H_pca + np.random.normal(0, 0.02, X_H_pca.shape)
    X_F_noisy = X_F_pca + np.random.normal(0, 0.02, X_F_pca.shape)
    
    # --- PQKR ---
    pqkr = PQKR(n_qubits=n_qubits, n_layers=2, seed=s)
    
    K_HH = pqkr.compute_kernel(X_H_noisy, X_H_noisy)
    K_FF = pqkr.compute_kernel(X_F_noisy, X_F_noisy)
    K_HF = pqkr.compute_kernel(X_H_noisy, X_F_noisy)
    
    # Eigenspectrum
    eigs_qH = np.real(np.linalg.eigvals(K_HH))
    eigs_qH.sort()
    q_spectra_H.append(eigs_qH[::-1])
    
    eigs_qF = np.real(np.linalg.eigvals(K_FF))
    eigs_qF.sort()
    q_spectra_F.append(eigs_qF[::-1])
    
    frob = frobenius_divergence(K_HH, K_FF)
    mmd = compute_mmd(K_HH, K_FF, K_HF)
    
    # Similarity
    intra_H = (np.sum(K_HH) - np.trace(K_HH)) / (num_windows*(num_windows-1))
    intra_F = (np.sum(K_FF) - np.trace(K_FF)) / (num_windows*(num_windows-1))
    inter = np.sum(K_HF) / (num_windows*num_windows)
    sep_ratio = compute_separation_ratio(intra_H, intra_F, inter)
    
    # Store
    q_frob_list.append(frob)
    q_mmd_list.append(mmd)
    q_intra_H_list.append(intra_H)
    q_intra_F_list.append(intra_F)
    q_inter_list.append(inter)
    q_sep_ratio_list.append(sep_ratio)
    
    seed_metrics.append({
        "seed": s,
        "frobenius": frob,
        "mmd": mmd,
        "intra_similarity": (intra_H + intra_F) / 2.0,  # Averaged intra-similarity for the table
        "inter_similarity": inter,
        "separation_ratio": sep_ratio
    })
    
    # Embedding metrics
    K_comb_q = np.vstack([np.hstack([K_HH, K_HF]), np.hstack([K_HF.T, K_FF])])
    dist_q = np.clip(1.0 - K_comb_q, 0, 1)
    
    psi_H = np.array([pqkr.get_state(x) for x in X_H_pca])
    psi_F = np.array([pqkr.get_state(x) for x in X_F_pca])
    psi_comb = np.vstack([np.hstack([np.real(psi_H), np.imag(psi_H)]), np.hstack([np.real(psi_F), np.imag(psi_F)])])
    
    q_sil_list.append(silhouette_score(dist_q, labels, metric='precomputed'))
    q_db_list.append(davies_bouldin_score(psi_comb, labels))
    q_ch_list.append(calinski_harabasz_score(psi_comb, labels))
    
    # --- Classical RBF ---
    gamma_val = 1.0 / n_qubits
    C_HH = rbf_kernel(X_H_noisy, X_H_noisy, gamma=gamma_val)
    C_FF = rbf_kernel(X_F_noisy, X_F_noisy, gamma=gamma_val)
    C_HF = rbf_kernel(X_H_noisy, X_F_noisy, gamma=gamma_val)
    
    eigs_cH = np.real(np.linalg.eigvals(C_HH))
    eigs_cH.sort()
    c_spectra_H.append(eigs_cH[::-1])
    
    c_frob = frobenius_divergence(C_HH, C_FF)
    c_mmd = compute_mmd(C_HH, C_FF, C_HF)
    
    c_intra_H = (np.sum(C_HH) - np.trace(C_HH)) / (num_windows*(num_windows-1))
    c_intra_F = (np.sum(C_FF) - np.trace(C_FF)) / (num_windows*(num_windows-1))
    c_inter = np.sum(C_HF) / (num_windows*num_windows)
    c_sep = compute_separation_ratio(c_intra_H, c_intra_F, c_inter)
    
    c_frob_list.append(c_frob)
    c_mmd_list.append(c_mmd)
    c_intra_H_list.append(c_intra_H)
    c_intra_F_list.append(c_intra_F)
    c_inter_list.append(c_inter)
    c_sep_ratio_list.append(c_sep)
    C_comb_c = np.vstack([np.hstack([C_HH, C_HF]), np.hstack([C_HF.T, C_FF])])
    dist_c = np.clip(1.0 - C_comb_c, 0, 1)

    c_sil_list.append(silhouette_score(dist_c, labels, metric='precomputed'))
    c_db_list.append(davies_bouldin_score(X_combined_pca, labels))
    c_ch_list.append(calinski_harabasz_score(X_combined_pca, labels))

df_seed_metrics = pd.DataFrame(seed_metrics)
df_seed_metrics.to_csv(os.path.join(results_tables, "pqkr_seed_metrics.csv"), index=False)


# TASK 2: Statistical Aggregation Layer
summary_data = {
    "Metric": ["Frobenius Divergence", "MMD", "Intra Similarity (H)", "Intra Similarity (F)", "Inter Similarity", "Separation Ratio", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"],
    "Quantum_Mean": [np.mean(q_frob_list), np.mean(q_mmd_list), np.mean(q_intra_H_list), np.mean(q_intra_F_list), np.mean(q_inter_list), np.mean(q_sep_ratio_list), np.mean(q_sil_list), np.mean(q_db_list), np.mean(q_ch_list)],
    "Quantum_Std": [np.std(q_frob_list), np.std(q_mmd_list), np.std(q_intra_H_list), np.std(q_intra_F_list), np.std(q_inter_list), np.std(q_sep_ratio_list), np.std(q_sil_list), np.std(q_db_list), np.std(q_ch_list)],
    "Quantum_95CI": [calc_ci(q_frob_list), calc_ci(q_mmd_list), calc_ci(q_intra_H_list), calc_ci(q_intra_F_list), calc_ci(q_inter_list), calc_ci(q_sep_ratio_list), calc_ci(q_sil_list), calc_ci(q_db_list), calc_ci(q_ch_list)],
    
    "Classical_Mean": [np.mean(c_frob_list), np.mean(c_mmd_list), np.mean(c_intra_H_list), np.mean(c_intra_F_list), np.mean(c_inter_list), np.mean(c_sep_ratio_list), np.mean(c_sil_list), np.mean(c_db_list), np.mean(c_ch_list)],
    "Classical_Std": [np.std(c_frob_list), np.std(c_mmd_list), np.std(c_intra_H_list), np.std(c_intra_F_list), np.std(c_inter_list), np.std(c_sep_ratio_list), np.std(c_sil_list), np.std(c_db_list), np.std(c_ch_list)],
    "Classical_95CI": [calc_ci(c_frob_list), calc_ci(c_mmd_list), calc_ci(c_intra_H_list), calc_ci(c_intra_F_list), calc_ci(c_inter_list), calc_ci(c_sep_ratio_list), calc_ci(c_sil_list), calc_ci(c_db_list), calc_ci(c_ch_list)]
}

# TASK 3: Significance Testing
# T-Test for MMD and Frob
try:
    _, p_mmd = ttest_ind(q_mmd_list, c_mmd_list)
    _, p_frob = ttest_ind(q_frob_list, c_frob_list)
except:
    p_mmd, p_frob = np.nan, np.nan

# KS-test between eigenvalue spectra (Healthy vs Fault for Quantum)
q_spectra_H_mean = np.mean(np.array(q_spectra_H), axis=0)
q_spectra_F_mean = np.mean(np.array(q_spectra_F), axis=0)
try:
    _, p_ks = ks_2samp(q_spectra_H_mean, q_spectra_F_mean)
except:
    p_ks = np.nan

p_values = [p_frob, p_mmd, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
summary_data["p_value_vs_classical"] = p_values

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(os.path.join(results_tables, "pqkr_summary_statistics.csv"), index=False)

# Recreate matrices for seed 0 to use in heatmaps
pqkr0 = PQKR(n_qubits=n_qubits, n_layers=2, seed=0)
K_HH0 = pqkr0.compute_kernel(X_H_pca, X_H_pca)
K_FF0 = pqkr0.compute_kernel(X_F_pca, X_F_pca)
K_HF0 = pqkr0.compute_kernel(X_H_pca, X_F_pca)

intra_H0 = (np.sum(K_HH0) - np.trace(K_HH0)) / (num_windows*(num_windows-1))
intra_F0 = (np.sum(K_FF0) - np.trace(K_FF0)) / (num_windows*(num_windows-1))
inter_HF0 = np.sum(K_HF0) / (num_windows*num_windows)
sep_ratio0 = compute_separation_ratio(intra_H0, intra_F0, inter_HF0)

# TASK 4: Heatmap Upgrade
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

im1 = axs[0].imshow(K_HH0, cmap='viridis', aspect='auto')
axs[0].set_title(f"Healthy Intra-Class\n$\\mu_{{intra}}={intra_H0:.3f}$, Sep Ratio={sep_ratio0:.2f}")
plt.colorbar(im1, ax=axs[0], label=r"Fidelity $|\langle\psi_i | \psi_j\rangle|^2$")
axs[0].set_xlabel("Window Index")
axs[0].set_ylabel("Window Index")

im2 = axs[1].imshow(K_FF0, cmap='viridis', aspect='auto')
axs[1].set_title(f"Fault Intra-Class\n$\\mu_{{intra}}={intra_F0:.3f}$")
plt.colorbar(im2, ax=axs[1], label=r"Fidelity $|\langle\psi_i | \psi_j\rangle|^2$")
axs[1].set_xlabel("Window Index")
axs[1].set_ylabel("Window Index")

plt.suptitle(f"Quantum Fidelity Kernel Matrices (Inter-Class $\\mu_{{inter}}={inter_HF0:.3f}$)", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(results_plots, "quantum_kernel_heatmaps_q1.png"), dpi=300)
plt.close()

# TASK 5: Replace 3D Cross-Fidelity Plot 
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(K_HF0, cmap='magma', aspect='auto')
ax.set_title(f"Structured 2D Cross-Fidelity Manifold (Healthy vs Fault)\n$\\mu_{{inter}}={inter_HF0:.3f}$, Sep Ratio={sep_ratio0:.2f}")
ax.set_xlabel("Fault Window Index")
ax.set_ylabel("Healthy Window Index")
plt.colorbar(im, ax=ax, label=r"Cross-Fidelity $|\langle\psi_x | \psi_y\rangle|^2$")
plt.tight_layout()
plt.savefig(os.path.join(results_plots, "cross_kernel_structure.png"), dpi=300)
plt.close()

# TASK 7: Eigen Spectrum Hardening
all_spectra_H = np.array(q_spectra_H)
all_spectra_F = np.array(q_spectra_F)

mean_spec_H = np.mean(all_spectra_H, axis=0)
std_spec_H = np.std(all_spectra_H, axis=0)
mean_spec_F = np.mean(all_spectra_F, axis=0)
std_spec_F = np.std(all_spectra_F, axis=0)

cum_energy_H = np.cumsum(mean_spec_H) / np.sum(mean_spec_H)
cum_energy_F = np.cumsum(mean_spec_F) / np.sum(mean_spec_F)

# Effective rank: rank index where cumulative energy > 95%
rank_H = np.argmax(cum_energy_H > 0.95) + 1
rank_F = np.argmax(cum_energy_F > 0.95) + 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(np.log10(mean_spec_H + 1e-12), label=f'Healthy (Eff. Rank={rank_H})', color='green')
ax1.fill_between(range(len(mean_spec_H)), np.log10(mean_spec_H - std_spec_H + 1e-12), np.log10(mean_spec_H + std_spec_H + 1e-12), color='green', alpha=0.2)
ax1.plot(np.log10(mean_spec_F + 1e-12), label=f'Fault (Eff. Rank={rank_F})', color='red')
ax1.fill_between(range(len(mean_spec_F)), np.log10(mean_spec_F - std_spec_F + 1e-12), np.log10(mean_spec_F + std_spec_F + 1e-12), color='red', alpha=0.2)
ax1.set_title(f"Quantum Kernel Log-Eigenvalue Decay\nKS-Test p={p_ks:.2e}")
ax1.set_xlabel("Eigenvalue Index")
ax1.set_ylabel(r"$\log_{10}(\lambda)$")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(cum_energy_H, label='Healthy', color='green')
ax2.plot(cum_energy_F, label='Fault', color='red')
ax2.axhline(0.95, color='black', linestyle='--', label='95% Energy')
ax2.set_title("Cumulative Spectral Energy")
ax2.set_xlabel("Eigenvalue Index")
ax2.set_ylabel("Cumulative Explained Variance")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(results_plots, "kernel_eigenspectrum_q1.png"), dpi=300)
plt.close()


# TASK 8: Quantum Advantage Table
adv_data = []

def format_cell(mean, std):
    return f"{mean:.4f} \u00B1 {std:.4f}"

idx_frob = df_summary.index[df_summary['Metric'] == 'Frobenius Divergence'][0]
idx_mmd = df_summary.index[df_summary['Metric'] == 'MMD'][0]
idx_sil = df_summary.index[df_summary['Metric'] == 'Silhouette'][0]
idx_sep = df_summary.index[df_summary['Metric'] == 'Separation Ratio'][0]

def pct_change(idx, lower_is_better=False):
    c_mean = df_summary.at[idx, 'Classical_Mean']
    q_mean = df_summary.at[idx, 'Quantum_Mean']
    if lower_is_better:
        return ((c_mean - q_mean) / c_mean) * 100
    else:
        return ((q_mean - c_mean) / c_mean) * 100

metrics_to_include = [
    ("Frobenius Divergence", idx_frob, True),
    ("MMD", idx_mmd, False),
    ("Silhouette", idx_sil, False),
    ("Separation Ratio", idx_sep, False)
]

for name, idx, lower_is_better in metrics_to_include:
    c_m = df_summary.at[idx, 'Classical_Mean']
    c_s = df_summary.at[idx, 'Classical_Std']
    q_m = df_summary.at[idx, 'Quantum_Mean']
    q_s = df_summary.at[idx, 'Quantum_Std']
    
    pct = pct_change(idx, lower_is_better)
    
    adv_data.append({
        "Metric": name,
        "Classical": format_cell(c_m, c_s),
        "Quantum": format_cell(q_m, q_s),
        "% Improvement": f"{pct:+.2f}%"
    })

df_adv = pd.DataFrame(adv_data)
df_adv.to_csv(os.path.join(results_tables, "quantum_advantage_table.csv"), index=False)

print("\nPHASE 3 HARDENING COMPLETE \u2014 READY FOR REVIEW")
