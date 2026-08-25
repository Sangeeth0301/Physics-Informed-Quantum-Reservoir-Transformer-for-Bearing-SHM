"""
Experiment: does switching PQKR's readout from a full-statevector fidelity
kernel to a true projected quantum kernel (Huang et al. 2021) change the
quantum-vs-classical separability comparison found in the project's own
Phase-3 hardening script (07_phase3_hardening.py)?

Methodology mirrors 07_phase3_hardening.py exactly (same data, same
num_windows=20, same seeds 0-9, same PCA-to-5, same Gaussian noise std=0.02,
same metrics) so the three columns (Classical RBF / Quantum Fidelity /
Quantum Projected) are a fair, apples-to-apples comparison.
"""
import sys, os, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import ttest_ind

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.data_prep.signal_processing import preprocess_bearing_signal
from pydmd import MrDMD, DMD
from src.quantum.pqkr import PQKR
from src.quantum.metrics import compute_mmd, frobenius_divergence

base_dir = os.path.abspath(os.path.dirname(__file__))
processed_dir = os.path.join(base_dir, 'data', 'processed')

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
        magnitudes = np.abs(eigs); reals = np.real(eigs); imags = np.imag(eigs)
        sorted_idx = np.lexsort((-imags, -reals, -magnitudes))
        top_eigs = eigs[sorted_idx][:4]
    else:
        top_eigs = []
    for _ in range(4 - len(top_eigs)):
        top_eigs = np.append(top_eigs, 0.0 + 0.0j)
    for e in top_eigs:
        feats.extend([np.real(e), np.imag(e), np.abs(e)])
    return np.array(feats, dtype=float)

print("[1] Loading CWRU processed windows...")
healthy_windows = np.load(os.path.join(processed_dir, "healthy_windows.npy"))
fault_windows = np.load(os.path.join(processed_dir, "fault_windows.npy"))

num_windows = min(20, len(fault_windows), len(healthy_windows))
print(f"    num_windows = {num_windows}")

print("[2] Extracting mrDMD/Koopman features (identical to 07_phase3_hardening.py)...")
features_H = np.array([extract_features(healthy_windows[i]) for i in range(num_windows)])
features_F = np.array([extract_features(fault_windows[i]) for i in range(num_windows)])

scaler = StandardScaler()
X_H_norm = scaler.fit_transform(features_H)
X_F_norm = scaler.transform(features_F)

n_qubits = 5
pca = PCA(n_components=n_qubits, random_state=42)
X_H_pca = pca.fit_transform(X_H_norm)
X_F_pca = pca.transform(X_F_norm)
labels = np.array([0] * num_windows + [1] * num_windows)

def compute_separation_ratio(intra_H, intra_F, inter_HF):
    return (intra_H + intra_F) / (2.0 * inter_HF + 1e-10)

def summarize(K_HH, K_FF, K_HF, dist_metric_matrix=None):
    frob = frobenius_divergence(K_HH, K_FF)
    mmd = compute_mmd(K_HH, K_FF, K_HF)
    intra_H = (np.sum(K_HH) - np.trace(K_HH)) / (num_windows * (num_windows - 1))
    intra_F = (np.sum(K_FF) - np.trace(K_FF)) / (num_windows * (num_windows - 1))
    inter = np.sum(K_HF) / (num_windows * num_windows)
    sep = compute_separation_ratio(intra_H, intra_F, inter)
    K_comb = np.vstack([np.hstack([K_HH, K_HF]), np.hstack([K_HF.T, K_FF])])
    dist = np.clip(1.0 - K_comb, 0, None)
    sil = silhouette_score(dist, labels, metric='precomputed')
    return dict(frob=frob, mmd=mmd, intra_H=intra_H, intra_F=intra_F, inter=inter, sep=sep, sil=sil)

seeds = list(range(10))
results = {"classical": [], "fidelity": [], "projected": []}

print("[3] Running 10-seed comparison: Classical RBF vs Quantum Fidelity vs Quantum Projected...")
for s in seeds:
    np.random.seed(s)
    X_H_noisy = X_H_pca + np.random.normal(0, 0.02, X_H_pca.shape)
    X_F_noisy = X_F_pca + np.random.normal(0, 0.02, X_F_pca.shape)

    # --- Classical RBF (identical to original script) ---
    gamma_val = 1.0 / n_qubits
    C_HH = rbf_kernel(X_H_noisy, X_H_noisy, gamma=gamma_val)
    C_FF = rbf_kernel(X_F_noisy, X_F_noisy, gamma=gamma_val)
    C_HF = rbf_kernel(X_H_noisy, X_F_noisy, gamma=gamma_val)
    results["classical"].append(summarize(C_HH, C_FF, C_HF))

    # --- Quantum Fidelity (original PQKR, unmodified) ---
    pqkr = PQKR(n_qubits=n_qubits, n_layers=2, seed=s)
    K_HH = pqkr.compute_kernel(X_H_noisy, X_H_noisy)
    K_FF = pqkr.compute_kernel(X_F_noisy, X_F_noisy)
    K_HF = pqkr.compute_kernel(X_H_noisy, X_F_noisy)
    results["fidelity"].append(summarize(K_HH, K_FF, K_HF))

    # --- Quantum Projected (new readout, same reservoir circuit/seed) ---
    P_HH = pqkr.compute_projected_kernel(X_H_noisy, X_H_noisy)
    P_FF = pqkr.compute_projected_kernel(X_F_noisy, X_F_noisy)
    P_HF = pqkr.compute_projected_kernel(X_H_noisy, X_F_noisy)
    results["projected"].append(summarize(P_HH, P_FF, P_HF))

    print(f"    seed {s}: classical sep={results['classical'][-1]['sep']:.3f}  "
          f"fidelity sep={results['fidelity'][-1]['sep']:.3f}  "
          f"projected sep={results['projected'][-1]['sep']:.3f}")

def agg(key, metric):
    vals = [r[metric] for r in results[key]]
    return float(np.mean(vals)), float(np.std(vals)), vals

print("\n[4] Summary (mean +/- std across 10 seeds):\n")
summary = {}
for metric in ["frob", "mmd", "sep", "sil"]:
    row = {}
    for key in ["classical", "fidelity", "projected"]:
        m, sd, vals = agg(key, metric)
        row[key] = {"mean": m, "std": sd, "vals": vals}
    # significance: projected vs classical, fidelity vs classical
    _, p_proj_vs_c = ttest_ind(row["projected"]["vals"], row["classical"]["vals"])
    _, p_fid_vs_c = ttest_ind(row["fidelity"]["vals"], row["classical"]["vals"])
    row["p_fidelity_vs_classical"] = float(p_fid_vs_c)
    row["p_projected_vs_classical"] = float(p_proj_vs_c)
    summary[metric] = row
    print(f"{metric:6s}  classical={row['classical']['mean']:.4f}±{row['classical']['std']:.4f}   "
          f"fidelity={row['fidelity']['mean']:.4f}±{row['fidelity']['std']:.4f} (p={p_fid_vs_c:.2e})   "
          f"projected={row['projected']['mean']:.4f}±{row['projected']['std']:.4f} (p={p_proj_vs_c:.2e})")

with open(os.path.join(base_dir, "results", "projected_kernel_experiment.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved -> results/projected_kernel_experiment.json")
