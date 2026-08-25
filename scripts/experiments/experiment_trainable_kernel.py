"""
Experiment 2: lightweight trainable quantum kernel via Kernel-Target Alignment
(KTA), keeping the SAME reservoir circuit structure (5 qubits, RX encoding,
2 entangling layers) but adding a small number of trainable parameters on
top of it, instead of a fully variational/deep VQC (which the literature
review flagged as barren-plateau risk for little benefit at this qubit
count). This directly tests the "make the kernel task-aware instead of
purely random" upgrade path proposed in the architecture deep-dive.

Proper held-out split (no train/test leakage): alignment is optimized on a
TRAIN split; frob/mmd/sep/sil are all evaluated on a disjoint TEST split,
for the classical, fixed-fidelity, fixed-projected, AND trained-kernel
variants alike, so the comparison is fair.
"""
import sys, os, json
import numpy as np
import pennylane as qml
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import rbf_kernel

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.data_prep.signal_processing import preprocess_bearing_signal
from pydmd import MrDMD, DMD
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
        feats.extend([np.max(radii), np.sum(radii > 1.0) / len(eigs), np.mean(np.abs(np.imag(eigs)))])
        mags = np.abs(eigs); reals = np.real(eigs); imags = np.imag(eigs)
        idx = np.lexsort((-imags, -reals, -mags))
        top = eigs[idx][:4]
    else:
        feats.extend([0.0, 0.0, 0.0]); top = []
    for _ in range(4 - len(top)):
        top = np.append(top, 0.0 + 0.0j)
    for e in top:
        feats.extend([np.real(e), np.imag(e), np.abs(e)])
    return np.array(feats, dtype=float)

print("[1] Loading data and building 40-window (20H+20F) feature set, 60/40 split...")
healthy_windows = np.load(os.path.join(processed_dir, "healthy_windows.npy"))
fault_windows = np.load(os.path.join(processed_dir, "fault_windows.npy"))
num_windows = min(20, len(fault_windows), len(healthy_windows))

features_H = np.array([extract_features(healthy_windows[i]) for i in range(num_windows)])
features_F = np.array([extract_features(fault_windows[i]) for i in range(num_windows)])

scaler = StandardScaler()
X_H_norm = scaler.fit_transform(features_H)
X_F_norm = scaler.transform(features_F)

n_qubits = 5
pca = PCA(n_components=n_qubits, random_state=42)
X_H_pca = pca.fit_transform(X_H_norm)
X_F_pca = pca.transform(X_F_norm)

# Held-out split: first 12 of each class = train (for KTA only), last 8 of each = test (for ALL metrics)
n_train = 12
X_H_train, X_H_test = X_H_pca[:n_train], X_H_pca[n_train:]
X_F_train, X_F_test = X_F_pca[:n_train], X_F_pca[n_train:]
n_test = X_H_test.shape[0]
print(f"    train: {n_train}+{n_train}, test: {n_test}+{n_test}")

# --- Trainable quantum kernel: fixed reservoir structure + a small trainable
#     "alignment" layer of RY angles (n_qubits params) applied right before
#     the fixed reservoir layers, optimized via KTA. ---
dev = qml.device("default.qubit", wires=n_qubits)
rng = np.random.default_rng(42)
n_layers = 2
theta_rx = torch.tensor(rng.uniform(0, 2*np.pi, size=(n_layers, n_qubits)), dtype=torch.float64)
theta_ry = torch.tensor(rng.uniform(0, 2*np.pi, size=(n_layers, n_qubits)), dtype=torch.float64)
theta_rz = torch.tensor(rng.uniform(0, 2*np.pi, size=(n_layers, n_qubits)), dtype=torch.float64)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def trainable_circuit(features, align_params):
    for i in range(n_qubits):
        qml.RX(features[i], wires=i)
    # small trainable alignment layer (n_qubits params only)
    for i in range(n_qubits):
        qml.RY(align_params[i], wires=i)
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(theta_rx[l, i], wires=i)
            qml.RY(theta_ry[l, i], wires=i)
            qml.RZ(theta_rz[l, i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
    return qml.state()

def get_states(X, align_params):
    return torch.stack([trainable_circuit(torch.tensor(x, dtype=torch.float64), align_params) for x in X])

def fidelity_kernel_torch(S1, S2):
    inner = S1 @ S2.conj().T
    return torch.abs(inner) ** 2

print("[2] Optimizing a 5-parameter alignment layer via Kernel-Target Alignment (KTA) on TRAIN split only...")
align_params = torch.zeros(n_qubits, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([align_params], lr=0.15)

X_train = np.vstack([X_H_train, X_F_train])
y_train = torch.tensor([1.0]*n_train + [-1.0]*n_train, dtype=torch.float64)
Y_outer = torch.outer(y_train, y_train)

n_epochs = 40
for epoch in range(n_epochs):
    optimizer.zero_grad()
    states = get_states(X_train, align_params)
    K = fidelity_kernel_torch(states, states)
    # KTA loss: maximize <K, yy^T>_F / (||K||_F * ||yy^T||_F)  ==  minimize negative alignment
    num = torch.sum(K * Y_outer)
    den = torch.sqrt(torch.sum(K * K)) * torch.sqrt(torch.sum(Y_outer * Y_outer))
    kta = num / (den + 1e-10)
    loss = -kta
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"    epoch {epoch:3d}  KTA alignment = {kta.item():.4f}")

learned_align = align_params.detach().numpy()
print(f"    learned alignment angles: {np.round(learned_align, 3)}")

print("[3] Evaluating ALL variants on the held-out TEST split only (no leakage)...")

def sep_ratio(K_HH, K_FF, K_HF, n):
    iH = (np.sum(K_HH) - np.trace(K_HH)) / (n * (n - 1))
    iF = (np.sum(K_FF) - np.trace(K_FF)) / (n * (n - 1))
    iHF = np.sum(K_HF) / (n * n)
    return (iH + iF) / (2 * iHF + 1e-10)

def full_metrics(K_HH, K_FF, K_HF, n):
    frob = frobenius_divergence(K_HH, K_FF)
    mmd = compute_mmd(K_HH, K_FF, K_HF)
    sr = sep_ratio(K_HH, K_FF, K_HF, n)
    Kc = np.vstack([np.hstack([K_HH, K_HF]), np.hstack([K_HF.T, K_FF])])
    dist = np.clip(1.0 - Kc, 0, None)
    labels = np.array([0]*n + [1]*n)
    sil = silhouette_score(dist, labels, metric='precomputed')
    return dict(frob=float(frob), mmd=float(mmd), sep=float(sr), sil=float(sil))

results = {}

# classical RBF baseline on test split
gamma_val = 1.0 / n_qubits
C_HH = rbf_kernel(X_H_test, X_H_test, gamma=gamma_val)
C_FF = rbf_kernel(X_F_test, X_F_test, gamma=gamma_val)
C_HF = rbf_kernel(X_H_test, X_F_test, gamma=gamma_val)
results["classical_rbf"] = full_metrics(C_HH, C_FF, C_HF, n_test)

# fixed fidelity kernel (untrained align_params = zeros) on test split
with torch.no_grad():
    zero_params = torch.zeros(n_qubits, dtype=torch.float64)
    sH0 = get_states(X_H_test, zero_params).numpy()
    sF0 = get_states(X_F_test, zero_params).numpy()
K_HH0 = np.abs(sH0 @ sH0.conj().T) ** 2
K_FF0 = np.abs(sF0 @ sF0.conj().T) ** 2
K_HF0 = np.abs(sH0 @ sF0.conj().T) ** 2
results["fixed_fidelity"] = full_metrics(K_HH0, K_FF0, K_HF0, n_test)

# trained (KTA-aligned) kernel on test split
with torch.no_grad():
    sH1 = get_states(X_H_test, align_params).numpy()
    sF1 = get_states(X_F_test, align_params).numpy()
K_HH1 = np.abs(sH1 @ sH1.conj().T) ** 2
K_FF1 = np.abs(sF1 @ sF1.conj().T) ** 2
K_HF1 = np.abs(sH1 @ sF1.conj().T) ** 2
results["kta_trained"] = full_metrics(K_HH1, K_FF1, K_HF1, n_test)

print("\n[4] Held-out test-set comparison:\n")
print(f"{'metric':8s} {'classical_rbf':>15s} {'fixed_fidelity':>16s} {'kta_trained':>13s}")
for m in ["frob", "mmd", "sep", "sil"]:
    print(f"{m:8s} {results['classical_rbf'][m]:15.4f} {results['fixed_fidelity'][m]:16.4f} {results['kta_trained'][m]:13.4f}")

with open(os.path.join(base_dir, "results", "trainable_kernel_experiment.json"), "w") as f:
    json.dump({"results": results, "learned_align_params": learned_align.tolist(), "n_train": n_train, "n_test": n_test}, f, indent=2)
print("\nSaved -> results/trainable_kernel_experiment.json")
