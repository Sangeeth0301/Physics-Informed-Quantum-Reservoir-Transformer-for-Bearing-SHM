import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import norm, ttest_ind, ks_2samp
from scipy.spatial import ConvexHull
import umap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quantum.pqkr import PQKR
from src.quantum.metrics import compute_mmd, frobenius_divergence
from src.data_prep.signal_processing import preprocess_bearing_signal
from pydmd import MrDMD, DMD

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'plots'))
tables_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'tables'))
os.makedirs(results_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

print("=======================================================================")
print("PHASE 3B: EXECUTING Q1-GRADE STATISTICAL HARDENING (SEED STABILITY)    ")
print("=======================================================================\n")

# --- Feature Extraction Pipeline ---
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
        feats.extend([np.max(radii), np.sum(radii > 1.0) / len(radii), np.mean(np.abs(np.imag(eigs)))])
        
        magnitudes = np.abs(eigs)
        reals, imags = np.real(eigs), np.imag(eigs)
        sorted_idx = np.lexsort((-imags, -reals, -magnitudes))
        top_eigs = eigs[sorted_idx][:4]
    else:
        feats.extend([0.0, 0.0, 0.0])
        top_eigs = []
        
    for _ in range(4 - len(top_eigs)):
        top_eigs = np.append(top_eigs, 0.0 + 0.0j)
    
    for e in top_eigs:
        feats.extend([np.real(e), np.imag(e), np.abs(e)])
        
    return np.array(feats, dtype=float)

# Load data
processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
healthy_windows = np.load(os.path.join(processed_dir, "healthy_windows.npy"))
fault_windows   = np.load(os.path.join(processed_dir, "fault_windows.npy"))

num_windows = min(20, len(fault_windows), len(healthy_windows))
print(f"[*] Extracting statistical phase-space invariants ({num_windows} samples)...")

# --- 1. Phase-Space Attractor Entropy & Volume (Convex Hull) ---
def compute_phase_volume(signal, tau=20):
    x_t = signal[:-2*tau]
    x_t1 = signal[tau:-tau]
    x_t2 = signal[2*tau:]
    points = np.vstack((x_t, x_t1, x_t2)).T
    return ConvexHull(points).volume

vol_H = [compute_phase_volume(healthy_windows[i]) for i in range(num_windows)]
vol_F = [compute_phase_volume(fault_windows[i]) for i in range(num_windows)]
_, p_val_vol = ttest_ind(vol_H, vol_F)
print(f"  -> Healthy Phase Volume: {np.mean(vol_H):.2e} ± {np.std(vol_H):.2e}")
print(f"  -> Fault Phase Volume:   {np.mean(vol_F):.2e} ± {np.std(vol_F):.2e} (p = {p_val_vol:.1e})")

# Feature Processing
X_H_raw = np.array([extract_features(healthy_windows[i]) for i in range(num_windows)])
X_F_raw = np.array([extract_features(fault_windows[i]) for i in range(num_windows)])

scaler = StandardScaler()
X_H_norm = scaler.fit_transform(X_H_raw)
X_F_norm = scaler.transform(X_F_raw)

n_qubits = 5  # Recommended operating point
pca = PCA(n_components=n_qubits, random_state=42)
X_H_pca = pca.fit_transform(X_H_norm)
X_F_pca = pca.transform(X_F_norm)

# --- 2. MULTI-SEED STATISTICAL SEPARABILITY (>= 10 SEEDS) ---
print("\n[*] Executing Multi-Seed Quantum Hardware Simulations (n=10)...")
n_seeds = 10
quantum_frob, quantum_mmd, classical_frob, classical_mmd = [], [], [], []
q_intra_H_list, q_intra_F_list = [], []

from sklearn.metrics.pairwise import rbf_kernel

def CI(data):
    return 1.96 * np.std(data) / np.sqrt(len(data))

all_eigs_HH = []

for seed in range(n_seeds):
    pqkr = PQKR(n_qubits=n_qubits, n_layers=2, seed=seed)
    K_HH = pqkr.compute_kernel(X_H_pca, X_H_pca)
    K_FF = pqkr.compute_kernel(X_F_pca, X_F_pca)
    K_HF = pqkr.compute_kernel(X_H_pca, X_F_pca)
    
    quantum_mmd.append(compute_mmd(K_HH, K_FF, K_HF))
    quantum_frob.append(frobenius_divergence(K_HH, K_FF))
    
    q_intra_H_list.append((np.sum(K_HH) - np.trace(K_HH)) / (num_windows*(num_windows-1)))
    q_intra_F_list.append((np.sum(K_FF) - np.trace(K_FF)) / (num_windows*(num_windows-1)))
    
    if seed == 0:
        # Save structured 2D Annotated Heatmap
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(K_HH, cmap='viridis', annot=False, cbar=True, square=True)
        plt.title(f"Healthy Intra-class $\mu={q_intra_H_list[0]:.2f}$")
        
        plt.subplot(1, 2, 2)
        sns.heatmap(K_HF, cmap='magma', annot=False, cbar=True, square=True)
        plt.title(f"Cross-class Similarity $\mu={np.mean(K_HF):.2f}$")
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "q1_2D_fidelity_heatmap_annotated.png"), dpi=300)
        plt.close()
        
    eigs_hh = np.real(np.linalg.eigvals(K_HH))
    eigs_hh.sort()
    eigs_hh = eigs_hh[::-1]
    all_eigs_HH.append(eigs_hh)
    
    # Classical Baseline
    gamma_val = 1.0 / n_qubits
    C_HH = rbf_kernel(X_H_pca, X_H_pca, gamma=gamma_val)
    C_FF = rbf_kernel(X_F_pca, X_F_pca, gamma=gamma_val)
    C_HF = rbf_kernel(X_H_pca, X_F_pca, gamma=gamma_val)
    
    classical_mmd.append(compute_mmd(C_HH, C_FF, C_HF))
    classical_frob.append(frobenius_divergence(C_HH, C_FF))

# Statistics computing
_, p_val_mmd = ttest_ind(quantum_mmd, classical_mmd)
_, p_val_frob = ttest_ind(quantum_frob, classical_frob)

print(f"  -> Quantum MMD:   {np.mean(quantum_mmd):.4f} ± {CI(quantum_mmd):.4f} (95% CI)")
print(f"  -> Classical MMD: {np.mean(classical_mmd):.4f} ± {CI(classical_mmd):.4f} (95% CI)")

# --- 3. SPECTRUM LOG-DECAY & KS-TEST ---
print("\n[*] Exporting Log-Eigenvalue Decay Distributions & KS-Test...")
all_eigs_HH = np.array(all_eigs_HH)
mean_eigs_HH = np.mean(all_eigs_HH, axis=0)
std_eigs_HH = np.std(all_eigs_HH, axis=0)

# Computing Classical Eigen-Spectrum for KS Test
eigs_c_hh = np.real(np.linalg.eigvals(rbf_kernel(X_H_pca, X_H_pca, gamma=1.0/n_qubits)))
eigs_c_hh.sort()
eigs_c_hh = eigs_c_hh[::-1]

ks_stat, ks_pval = ks_2samp(mean_eigs_HH, eigs_c_hh)

plt.figure(figsize=(8, 6))
plt.plot(np.log10(mean_eigs_HH + 1e-10), color='blue', label='Quantum Kernel Spectrum')
plt.fill_between(range(len(mean_eigs_HH)), 
                 np.log10(mean_eigs_HH - std_eigs_HH + 1e-10), 
                 np.log10(mean_eigs_HH + std_eigs_HH + 1e-10), color='blue', alpha=0.2)

plt.plot(np.log10(eigs_c_hh + 1e-10), color='red', linestyle='--', label='Classical RBF Spectrum')
plt.title(f"Log-Eigenvalue Decay Distribution (KS-Test p-val: {ks_pval:.2e})", fontsize=14)
plt.ylabel(r"$\log_{10}(\lambda_i)$", fontsize=12)
plt.xlabel("Index $i$", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "q1_log_eigenvalue_decay_benched.png"), dpi=300)
plt.close()

# --- 4. UMAP QUANTUM EMBEDDINGS & MANIFOLD METRICS ---
print("\n[*] Generating UMAP 2D Density Mapping with Clustering Invariants...")
pqkr_base = PQKR(n_qubits=n_qubits, n_layers=2, seed=42)
psi_H = np.array([pqkr_base.get_state(x) for x in X_H_pca])
psi_F = np.array([pqkr_base.get_state(x) for x in X_F_pca])
psi_comb = np.vstack([np.abs(psi_H), np.abs(psi_F)])
labels = np.array([0]*num_windows + [1]*num_windows)

reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(psi_comb)

# Calculate unsupervised invariant metrics
sil_score = silhouette_score(embedding, labels)
db_score = davies_bouldin_score(embedding, labels)
print(f"  -> UMAP Silhouette Score: {sil_score:.4f} (Closer to 1 is better)")
print(f"  -> UMAP Davies-Bouldin:   {db_score:.4f} (Closer to 0 is better)")

plt.figure(figsize=(8, 6))
plt.scatter(embedding[labels==0, 0], embedding[labels==0, 1], c='green', s=60, alpha=0.8, edgecolor='k', label='Healthy $\mathcal{H}$')
plt.scatter(embedding[labels==1, 0], embedding[labels==1, 1], c='red', s=60, alpha=0.8, edgecolor='k', label='Fault $\mathcal{F}$')
plt.title(f"UMAP Quantum Projection Subspace (Sil={sil_score:.2f})", fontsize=14)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "q1_umap_quantum_projection.png"), dpi=300)
plt.close()

print("\n🚀 Q1-READY HARDENED STATISTICS COMPLETE AND EXPORTED.")
