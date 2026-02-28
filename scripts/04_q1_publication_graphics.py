import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

from src.quantum.pqkr import PQKR
from src.quantum.metrics import compute_mmd, frobenius_divergence
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'plots'))
tables_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'tables'))
os.makedirs(results_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

print("=========================================================================")
print("GENERATING Q1-GRADE QUANTUM TOPOLOGY AND HYPERPARAMETER ABLATION GRAPHICS")
print("=========================================================================\n")

# Re-defining feature extraction to keep this script robust
from src.data_prep.signal_processing import preprocess_bearing_signal
from pydmd import MrDMD, DMD

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
    except Exception:
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
        feats.append(np.real(e))
        feats.append(np.imag(e))
        feats.append(np.abs(e))
    return np.array(feats, dtype=float)

# Load data
processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
healthy_windows = np.load(os.path.join(processed_dir, "healthy_windows.npy"))
fault_windows   = np.load(os.path.join(processed_dir, "fault_windows.npy"))

num_windows = min(20, len(fault_windows), len(healthy_windows)) # Keep lightweight for memory
X_healthy_raw = np.array([extract_features(healthy_windows[i]) for i in range(num_windows)])
X_fault_raw = np.array([extract_features(fault_windows[i]) for i in range(num_windows)])

# Z-score Normalization
scaler = StandardScaler()
X_healthy_norm = scaler.fit_transform(X_healthy_raw)
X_fault_norm = scaler.transform(X_fault_raw)

# PCA
n_qubits = 4
pca = PCA(n_components=n_qubits, random_state=42)
X_H_pca = pca.fit_transform(X_healthy_norm)
X_F_pca = pca.transform(X_fault_norm)

# Generate Quantum States representing the Hilbert Space Map
print("[1] Calculating Deep Hilbert Space Topological Embeddings (t-SNE/UMAP)...")
pqkr = PQKR(n_qubits=n_qubits, n_layers=2, seed=42)
psi_H = np.array([pqkr.get_state(x) for x in X_H_pca])
psi_F = np.array([pqkr.get_state(x) for x in X_F_pca])

# Density unrolling: Map absolute complex waveforms down to 3D manifold
psi_combined = np.vstack([np.abs(psi_H), np.abs(psi_F)])
labels = np.array([0]*num_windows + [1]*num_windows)

tsne = TSNE(n_components=3, perplexity=10, random_state=42)
psi_tsne = tsne.fit_transform(psi_combined)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(psi_tsne[:num_windows, 0], psi_tsne[:num_windows, 1], psi_tsne[:num_windows, 2], 
           color='green', s=100, label='Healthy States', alpha=0.8, edgecolor='k')
ax.scatter(psi_tsne[num_windows:, 0], psi_tsne[num_windows:, 1], psi_tsne[num_windows:, 2], 
           color='red', s=100, label='Incipient Fault States', alpha=0.8, edgecolor='k')
ax.set_title("Quantum Hilbert Space Topology Unrolling (t-SNE)", fontsize=14)
ax.set_xlabel("Latent Dim 1")
ax.set_ylabel("Latent Dim 2")
ax.set_zlabel("Latent Dim 3")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "quantum_hilbert_space_manifold.png"), dpi=300)
plt.close()
print("  -> Saved quantum_hilbert_space_manifold.png")

print("\n[2] Generating 3D Continuous Quantum Fidelity Topography...")
Q_HF = pqkr.compute_kernel(X_H_pca, X_F_pca)

X_grid, Y_grid = np.meshgrid(np.arange(Q_HF.shape[1]), np.arange(Q_HF.shape[0]))
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X_grid, Y_grid, Q_HF, cmap='magma', linewidth=0, antialiased=True, alpha=0.9)
ax.set_title("3D Cross-Fidelity Manifold (Healthy vs Fault)", fontsize=14)
ax.set_xlabel("Fault Window Index")
ax.set_ylabel("Healthy Window Index")
ax.set_zlabel(r"Fidelity $|\langle\psi_i | \psi_j\rangle|^2$")
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "fidelity_3d_topography.png"), dpi=300)
plt.close()
print("  -> Saved fidelity_3d_topography.png")


print("\n[3] Generating Extensive Hyperparameter Ablation Tables (LaTeX/CSV)...")
qubit_range = [3, 4, 5, 6]
layer_range = [1, 2, 3]

results_grid = []

for q in qubit_range:
    pca_q = PCA(n_components=q, random_state=42)
    X_H_q = pca_q.fit_transform(X_healthy_norm)
    X_F_q = pca_q.transform(X_fault_norm)
    
    for l in layer_range:
        pqkr_test = PQKR(n_qubits=q, n_layers=l, seed=0)
        K_HH = pqkr_test.compute_kernel(X_H_q, X_H_q)
        K_FF = pqkr_test.compute_kernel(X_F_q, X_F_q)
        K_HF = pqkr_test.compute_kernel(X_H_q, X_F_q)
        
        mmd_val = compute_mmd(K_HH, K_FF, K_HF)
        frob_val = frobenius_divergence(K_HH, K_FF)
        intra_H = (np.sum(K_HH) - np.trace(K_HH)) / (num_windows*(num_windows-1))
        
        results_grid.append({
            "Qubits": q,
            "Layers": l,
            "Total Parameters": q * l * 3,
            "Frobenius Divergence": f"{frob_val:.4f}",
            "MMD": f"{mmd_val:.4f}",
            "Healthy Intra-Similarity": f"{intra_H:.4f}"
        })

df_grid = pd.DataFrame(results_grid)
df_grid.to_csv(os.path.join(tables_dir, "hyperparameter_ablation_grid.csv"), index=False)

with open(os.path.join(tables_dir, "hyperparameter_ablation_grid.tex"), "w") as f:
    f.write(df_grid.style.to_latex(
        caption="Extensive Hyperparameter Ablation: Circuit Depth and Dimensionality",
        label="tab:hyperparameter_grid",
        hrules=True
    ))
print("  -> Saved extensive ablation grid to results/tables/hyperparameter_ablation_grid.csv & .tex")

print("\n🚀 Q1 VISUALIZATIONS COMPLETE. Ready for Academic Review.")
