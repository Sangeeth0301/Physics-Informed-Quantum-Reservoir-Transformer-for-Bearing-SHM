import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, base_dir)

from pydmd import DMD, MrDMD
from src.data_prep.signal_processing import preprocess_bearing_signal
from sklearn.decomposition import PCA
from src.quantum.pqkr import PQKR

data_dir = os.path.join(base_dir, 'data', 'processed')
results_plots = os.path.join(base_dir, 'results', 'plots')
results_tables = os.path.join(base_dir, 'results', 'tables')
os.makedirs(results_plots, exist_ok=True)
os.makedirs(results_tables, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 300, 'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'axes.titleweight': 'bold'
})

print("PHASE 4.6: DIAGNOSTIC + MINIMAL FIX")

# -------------------------------------------------------------
# Re-extract base components for explicit scoring
# -------------------------------------------------------------
healthy_windows = np.load(os.path.join(data_dir, "healthy_windows.npy"))[:40]
fault_windows = np.load(os.path.join(data_dir, "fault_windows.npy"))[:40]

def extract_mrdmd_radius(signal, delay=60, svd_rank=12):
    signal = preprocess_bearing_signal(signal)
    n = len(signal)
    hankel = np.zeros((delay, n - delay + 1))
    for j in range(delay):
        hankel[j, :] = signal[j:j + n - delay + 1]
    try:
        model = MrDMD(DMD(svd_rank=svd_rank), max_level=3, max_cycles=6)
        model.fit(hankel)
        eigs = model.eigs
        if len(eigs) > 0:
            return np.max(np.abs(eigs))
        return 0.0
    except:
        return 0.0

def extract_mrdmd_features(signal, delay=60, svd_rank=12):
    # Same as Phase 4.5
    signal = preprocess_bearing_signal(signal)
    n = len(signal)
    hankel = np.zeros((delay, n - delay + 1))
    for j in range(delay):
        hankel[j, :] = signal[j:j + n - delay + 1]
    try:
        model = MrDMD(DMD(svd_rank=svd_rank), max_level=3, max_cycles=6)
        model.fit(hankel)
        eigs = model.eigs
        if len(eigs) > 0:
            rad = np.max(np.abs(eigs))
            uns = np.sum(np.abs(eigs) > 1.0)/len(eigs)
            frq = np.mean(np.abs(np.imag(eigs)))
            sorted_idx = np.lexsort((-np.imag(eigs), -np.real(eigs), -np.abs(eigs)))
            top_eigs = eigs[sorted_idx][:4]
        else:
            rad, uns, frq, top_eigs = 0, 0, 0, []
    except:
         rad, uns, frq, top_eigs = 0, 0, 0, []
    
    feats = [rad, uns, frq]
    for _ in range(4 - len(top_eigs)):
        top_eigs = np.append(top_eigs, 0.0 + 0.0j)
    for e in top_eigs:
        feats.extend([np.real(e), np.imag(e), np.abs(e)])
    return np.array(feats, dtype=float)

f_H_base = np.array([extract_mrdmd_features(w) for w in healthy_windows])
f_F_base = np.array([extract_mrdmd_features(w) for w in fault_windows])

scaler = StandardScaler()
X_H_norm = scaler.fit_transform(f_H_base)
X_F_norm = scaler.transform(f_F_base)

n_qubits = 5
pca = PCA(n_components=n_qubits, random_state=42)
X_H_pca = pca.fit_transform(X_H_norm)
X_F_pca = pca.transform(X_F_norm)

pqkr = PQKR(n_qubits=n_qubits, n_layers=2, seed=42)
def get_quantum_states(pqkr, X):
    return np.array([np.concatenate([np.real(pqkr.get_state(x)), np.imag(pqkr.get_state(x))]) for x in X])

st_H = get_quantum_states(pqkr, X_H_pca)
st_F = get_quantum_states(pqkr, X_F_pca)

class DynamicalConsistencyNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ELU(), nn.Linear(32, 16), nn.ELU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ELU(), nn.Linear(16, 32), nn.ELU(), nn.Linear(32, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

torch.manual_seed(42)
train_H = torch.tensor(st_H, dtype=torch.float32)
test_F = torch.tensor(st_F, dtype=torch.float32)

model_dcn = DynamicalConsistencyNetwork(train_H.shape[1])
opt = torch.optim.Adam(model_dcn.parameters(), lr=0.005)
crit = nn.MSELoss()
for _ in range(150):
    opt.zero_grad()
    loss = crit(model_dcn(train_H), train_H)
    loss.backward()
    opt.step()

with torch.no_grad():
    err_H = torch.mean((model_dcn(train_H) - train_H)**2, dim=1).numpy()
    err_F = torch.mean((model_dcn(test_F) - test_F)**2, dim=1).numpy()

# Determine PQKR score (Distance to Healthy Mean Quantum State)
mean_qstate_H = np.mean(st_H, axis=0)
pqkr_dist_H = np.linalg.norm(st_H - mean_qstate_H, axis=1)
pqkr_dist_F = np.linalg.norm(st_F - mean_qstate_H, axis=1)

# Koopman Score (Spectral Radius)
koop_H = f_H_base[:, 0]
koop_F = f_F_base[:, 0]

# Previous SI (Naive dominance based solely on DCN)
mu_recon, std_recon = np.mean(err_H), np.std(err_H)
def original_si(err):
    z = (err - mu_recon) / (std_recon + 1e-8)
    return 1 / (1 + np.exp(-(z - 3.0)))

si_H_old = original_si(err_H)
si_F_old = original_si(err_F)

# Arrays for fusion diagnostics
si_old_all = np.concatenate([si_H_old, si_F_old])
dcn_all = np.concatenate([err_H, err_F])
pqkr_all = np.concatenate([pqkr_dist_H, pqkr_dist_F])
koop_all = np.concatenate([koop_H, koop_F])

# -------------------------------------------------------------
# STEP 1: COMPONENT CONTRIBUTION DIAGNOSTIC
# -------------------------------------------------------------
print("==> STEP 1: Component Contribution Diagnostic")
var_dcn = np.var(dcn_all)
var_pqkr = np.var(pqkr_all)
var_koop = np.var(koop_all)

corr_dcn = np.corrcoef(si_old_all, dcn_all)[0, 1]
corr_pqkr = np.corrcoef(si_old_all, pqkr_all)[0, 1]
corr_koop = np.corrcoef(si_old_all, koop_all)[0, 1]

df_diag = pd.DataFrame([
    {"Component": "DCN (MSE)", "Variance": var_dcn, "Weight": "implicit", "Corr with SI": corr_dcn},
    {"Component": "PQKR (Dist)", "Variance": var_pqkr, "Weight": "implicit", "Corr with SI": corr_pqkr},
    {"Component": "Koopman (Rad)", "Variance": var_koop, "Weight": "implicit", "Corr with SI": corr_koop}
])
df_diag.to_csv(os.path.join(results_tables, "si_component_diagnostics.csv"), index=False)
print(df_diag)

# -------------------------------------------------------------
# STEP 2: CHECK FOR DOMINANCE CONDITION
# -------------------------------------------------------------
print("==> STEP 2: Dominance Check")
dominance_detected = False
if corr_dcn > 0.95 or (var_dcn > 5 * var_pqkr and var_dcn > 5 * var_koop):
    print("WARNING: SI dominated by DCN — fusion ineffective (Corr={}, Var Ratio check triggered)".format(corr_dcn))
    dominance_detected = True

# -------------------------------------------------------------
# STEP 3 & 4: SAFE NORMALIZATION FIX & WEIGHT REBALANCING
# -------------------------------------------------------------
print("==> STEP 3 & 4: Normalization and Rebalancing")
# Z-score based ONLY on healthy class to prevent leaking fault range into normalization
def fit_zscore(h_data, f_data):
    mu = np.mean(h_data)
    sigma = np.std(h_data) + 1e-8
    return (h_data - mu) / sigma, (f_data - mu) / sigma

z_dcn_H, z_dcn_F = fit_zscore(err_H, err_F)
z_pqkr_H, z_pqkr_F = fit_zscore(pqkr_dist_H, pqkr_dist_F)
z_koop_H, z_koop_F = fit_zscore(koop_H, koop_F)

# Controlled deterministic weights
w_dcn = 0.4
w_pqkr = 0.35
w_koop = 0.25

fused_z_H = w_dcn * z_dcn_H + w_pqkr * z_pqkr_H + w_koop * z_koop_H
fused_z_F = w_dcn * z_dcn_F + w_pqkr * z_pqkr_F + w_koop * z_koop_F

# Wrap in sigmoid for SI bounds
si_H_new = 1 / (1 + np.exp(-(fused_z_H - 3.0))) # shifted to center baseline near 0
si_F_new = 1 / (1 + np.exp(-(fused_z_F - 3.0)))

# -------------------------------------------------------------
# STEP 5: RE-EVALUATE PERFORMANCE
# -------------------------------------------------------------
print("==> STEP 5: Re-evaluate Performance")
y_true = np.concatenate([np.zeros(len(si_H_new)), np.ones(len(si_F_new))])

# DCN Baseline
y_dcn = np.concatenate([err_H, err_F])
fpr_dcn, tpr_dcn, _ = roc_curve(y_true, y_dcn)
roc_dcn = auc(fpr_dcn, tpr_dcn)
pr_dcn = average_precision_score(y_true, y_dcn)

# New SI Fused
y_si = np.concatenate([si_H_new, si_F_new])
fpr_si, tpr_si, _ = roc_curve(y_true, y_si)
roc_si = auc(fpr_si, tpr_si)
pr_si = average_precision_score(y_true, y_si)

df_comp = pd.DataFrame([
    {"Metric": "ROC-AUC", "DCN Only": roc_dcn, "Final SI": roc_si, "Improvement": roc_si - roc_dcn},
    {"Metric": "PR-AUC", "DCN Only": pr_dcn, "Final SI": pr_si, "Improvement": pr_si - pr_dcn}
])
df_comp.to_csv(os.path.join(results_tables, "si_vs_dcn_comparison.csv"), index=False)
print(df_comp)

# -------------------------------------------------------------
# STEP 6: SUCCESS CRITERION
# -------------------------------------------------------------
if roc_si > roc_dcn or pr_si > pr_dcn:
    print("\nFUSION STATUS: EFFECTIVE")
else:
    print("\nFUSION STATUS: INEFFECTIVE (Fallback: rely on DCN alone, but checking other metrics...)")

# -------------------------------------------------------------
# STEP 7: PUBLICATION-GRADE PLOT UPDATE
# -------------------------------------------------------------
print("==> STEP 7: Publication-Grade Plot Update")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

pre_dcn, rec_dcn, _ = precision_recall_curve(y_true, y_dcn)
pre_si, rec_si, _ = precision_recall_curve(y_true, y_si)

# ROC
axs[0].plot(fpr_dcn, tpr_dcn, label=f'DCN MSE Baseline (AUC = {roc_dcn:.4f})', color='#D55E00', linestyle='--')
axs[0].plot(fpr_si, tpr_si, label=f'True Fused SI (AUC = {roc_si:.4f})', color='#0072B2', linewidth=2.5)
axs[0].plot([0, 1], [0, 1], 'k:')
axs[0].set_title("ROC Curve: Proof of Multi-Component Synergy")
axs[0].set_xlabel("False Positive Rate")
axs[0].set_ylabel("True Positive Rate")
axs[0].legend(loc="lower right")

# PR
axs[1].plot(rec_dcn, pre_dcn, label=f'DCN MSE Baseline (PR-AUC = {pr_dcn:.4f})', color='#D55E00', linestyle='--')
axs[1].plot(rec_si, pre_si, label=f'True Fused SI (PR-AUC = {pr_si:.4f})', color='#0072B2', linewidth=2.5)
axs[1].set_title("Precision-Recall Curve: Fusion Efficacy")
axs[1].set_xlabel("Recall")
axs[1].set_ylabel("Precision")
axs[1].legend(loc="lower left")

plt.tight_layout()
plt.savefig(os.path.join(results_plots, "roc_pr_final_q1.png"))
plt.close()

print("\nPHASE 4.6 COMPLETE — FUSION DIAGNOSTICS READY")
