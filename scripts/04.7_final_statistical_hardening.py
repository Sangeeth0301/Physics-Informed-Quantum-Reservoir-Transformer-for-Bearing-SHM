import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, base_dir)

from pydmd import DMD, MrDMD
from src.data_prep.signal_processing import preprocess_bearing_signal
from src.quantum.pqkr import PQKR

data_dir = os.path.join(base_dir, 'data', 'processed')
results_plots = os.path.join(base_dir, 'results', 'plots', 'publication_ready')
results_tables = os.path.join(base_dir, 'results', 'tables')
os.makedirs(results_plots, exist_ok=True)
os.makedirs(results_tables, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 300, 'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'axes.titleweight': 'bold'
})

print("PART A: FINAL STATISTICAL HARDENING")

# 1. Base Setup (Same extraction strategy)
healthy_windows = np.load(os.path.join(data_dir, "healthy_windows.npy"))[:40]
fault_windows = np.load(os.path.join(data_dir, "fault_windows.npy"))[:40]

def extract_mrdmd_features(signal, delay=60, svd_rank=12):
    signal = preprocess_bearing_signal(signal)
    n = len(signal)
    hankel = np.zeros((delay, n - delay + 1))
    for j in range(delay): hankel[j, :] = signal[j:j + n - delay + 1]
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
    for _ in range(4 - len(top_eigs)): top_eigs = np.append(top_eigs, 0.0 + 0.0j)
    for e in top_eigs: feats.extend([np.real(e), np.imag(e), np.abs(e)])
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

class DynamicalConsistencyNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ELU(), nn.Linear(32, 16), nn.ELU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ELU(), nn.Linear(16, 32), nn.ELU(), nn.Linear(32, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

def get_quantum_states(pqkr, X):
    return np.array([np.concatenate([np.real(pqkr.get_state(x)), np.imag(pqkr.get_state(x))]) for x in X])

def fit_zscore(h_data, f_data):
    mu = np.mean(h_data)
    sigma = np.std(h_data) + 1e-8
    return (h_data - mu) / sigma, (f_data - mu) / sigma

# -------------------------------------------------------------------
# A1: Multi-Seed PR-AUC Statistics (10 Seeds)
# -------------------------------------------------------------------
print("==> A1: Multi-Seed PR-AUC Statistics")

y_true = np.concatenate([np.zeros(len(healthy_windows)), np.ones(len(fault_windows))])
seed_results = []
seeds = list(range(42, 52))

dcn_aucs = []
si_aucs = []

# Using fixed weights from Phase 4.6
w_dcn, w_pqkr, w_koop = 0.4, 0.35, 0.25

for s in seeds:
    pqkr = PQKR(n_qubits=n_qubits, n_layers=2, seed=s)
    st_H = get_quantum_states(pqkr, X_H_pca)
    st_F = get_quantum_states(pqkr, X_F_pca)
    
    torch.manual_seed(s)
    train_H = torch.tensor(st_H, dtype=torch.float32)
    test_F = torch.tensor(st_F, dtype=torch.float32)

    model_dcn = DynamicalConsistencyNetwork(train_H.shape[1])
    opt = torch.optim.Adam(model_dcn.parameters(), lr=0.005)
    crit = nn.MSELoss()
    for _ in range(100):
        opt.zero_grad()
        loss = crit(model_dcn(train_H), train_H)
        loss.backward()
        opt.step()
        
    with torch.no_grad():
        err_H = torch.mean((model_dcn(train_H) - train_H)**2, dim=1).numpy()
        err_F = torch.mean((model_dcn(test_F) - test_F)**2, dim=1).numpy()
    
    mean_qstate_H = np.mean(st_H, axis=0)
    pqkr_dist_H = np.linalg.norm(st_H - mean_qstate_H, axis=1)
    pqkr_dist_F = np.linalg.norm(st_F - mean_qstate_H, axis=1)
    
    koop_H = f_H_base[:, 0]
    koop_F = f_F_base[:, 0]
    
    z_dcn_H, z_dcn_F = fit_zscore(err_H, err_F)
    z_pqkr_H, z_pqkr_F = fit_zscore(pqkr_dist_H, pqkr_dist_F)
    z_koop_H, z_koop_F = fit_zscore(koop_H, koop_F)
    
    fused_z_H = w_dcn * z_dcn_H + w_pqkr * z_pqkr_H + w_koop * z_koop_H
    fused_z_F = w_dcn * z_dcn_F + w_pqkr * z_pqkr_F + w_koop * z_koop_F
    
    si_H_new = 1 / (1 + np.exp(-(fused_z_H - 3.0)))
    si_F_new = 1 / (1 + np.exp(-(fused_z_F - 3.0)))
    
    y_dcn = np.concatenate([err_H, err_F])
    pr_dcn = average_precision_score(y_true, y_dcn)
    
    y_si = np.concatenate([si_H_new, si_F_new])
    pr_si = average_precision_score(y_true, y_si)
    
    dcn_aucs.append(pr_dcn)
    si_aucs.append(pr_si)
    
    seed_results.append({"Seed": s, "PR_AUC_DCN": pr_dcn, "PR_AUC_SI": pr_si})

df_seeds = pd.DataFrame(seed_results)
df_seeds.to_csv(os.path.join(results_tables, "pr_auc_seed_stats_raw.csv"), index=False)

def get_stats(arr):
    m = np.mean(arr)
    s = np.std(arr)
    ci = 1.96 * s / np.sqrt(len(arr))
    return m, s, ci

m_dcn, s_dcn, ci_dcn = get_stats(dcn_aucs)
m_si, s_si, ci_si = get_stats(si_aucs)

pd.DataFrame([
    {"Model": "DCN Baseline", "Mean_PR_AUC": m_dcn, "Std": s_dcn, "95_CI": ci_dcn},
    {"Model": "Final Fused SI", "Mean_PR_AUC": m_si, "Std": s_si, "95_CI": ci_si}
]).to_csv(os.path.join(results_tables, "pr_auc_seed_stats.csv"), index=False)
print("Stat aggregation saved.")

# -------------------------------------------------------------------
# A2: Paired Statistical Test
# -------------------------------------------------------------------
print("==> A2: Paired Statistical Test")
t_stat, p_val = ttest_rel(si_aucs, dcn_aucs, alternative='greater')
is_sig = p_val < 0.05
pd.DataFrame([{
    "Hypothesis Test": "Paired T-test (PR_AUC_SI > PR_AUC_DCN)",
    "T-Statistic": t_stat,
    "p-value": f"{p_val:.4e}",
    "Significant (p<0.05)": is_sig
}]).to_csv(os.path.join(results_tables, "si_vs_dcn_significance.csv"), index=False)
print(f"Paired T-Test: t={t_stat:.4f}, p={p_val:.4e} -> Significant: {is_sig}")

# -------------------------------------------------------------------
# A3: Fusion Dominance Check
# -------------------------------------------------------------------
print("==> A3: Fusion Dominance Check")
# Recompute for default seed 42 to get the full arrays
si_all = np.concatenate([si_H_new, si_F_new])
dcn_var = np.var(fused_z_H) # not quite right, let's look at the arrays directly
z_dcn_all = np.concatenate([z_dcn_H, z_dcn_F])
z_pqkr_all = np.concatenate([z_pqkr_H, z_pqkr_F])
z_koop_all = np.concatenate([z_koop_H, z_koop_F])

var_dcn = np.var(w_dcn * z_dcn_all)
var_pqkr = np.var(w_pqkr * z_pqkr_all)
var_koop = np.var(w_koop * z_koop_all)

corr_dcn = np.corrcoef(si_all, z_dcn_all)[0, 1]
corr_pqkr = np.corrcoef(si_all, z_pqkr_all)[0, 1]
corr_koop = np.corrcoef(si_all, z_koop_all)[0, 1]

pd.DataFrame([
    {"Component": "DCN_MSE (Normalized)", "Scaled_Variance": var_dcn, "Corr_with_SI_Final": corr_dcn},
    {"Component": "PQKR_Dist (Normalized)", "Scaled_Variance": var_pqkr, "Corr_with_SI_Final": corr_pqkr},
    {"Component": "Koopman_Rad (Normalized)", "Scaled_Variance": var_koop, "Corr_with_SI_Final": corr_koop},
]).to_csv(os.path.join(results_tables, "fusion_component_analysis.csv"), index=False)
print("Fusion dominance check complete.")

# -------------------------------------------------------------------
# A4: Figure Polish (NON-DESTRUCTIVE)
# -------------------------------------------------------------------
print("==> A4: Figure Polish")
fpr_dcn, tpr_dcn, _ = roc_curve(y_true, y_dcn)
roc_dcn = auc(fpr_dcn, tpr_dcn)
fpr_si, tpr_si, _ = roc_curve(y_true, y_si)
roc_si = auc(fpr_si, tpr_si)

pre_dcn, rec_dcn, _ = precision_recall_curve(y_true, y_dcn)
pre_si, rec_si, _ = precision_recall_curve(y_true, y_si)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(fpr_dcn, tpr_dcn, label=f'DCN Baseline (AUC = {roc_dcn:.4f})', color='#D55E00', linestyle='--', linewidth=2)
axs[0].plot(fpr_si, tpr_si, label=f'Fused Model SI (AUC = {roc_si:.4f})', color='#0072B2', linewidth=2.5)
axs[0].plot([0, 1], [0, 1], 'k:')
axs[0].set_title("ROC Curve Comparison")
axs[0].set_xlabel("False Positive Rate")
axs[0].set_ylabel("True Positive Rate")
axs[0].legend(loc="lower right")

axs[1].plot(rec_dcn, pre_dcn, label=f'DCN Baseline (PR-AUC = {pr_dcn:.4f})', color='#D55E00', linestyle='--', linewidth=2)
axs[1].plot(rec_si, pre_si, label=f'Fused Model SI (PR-AUC = {pr_si:.4f})', color='#0072B2', linewidth=2.5)
axs[1].set_title("Precision-Recall Curve Comparison")
axs[1].set_xlabel("Recall")
axs[1].set_ylabel("Precision")
axs[1].legend(loc="lower left")

plt.tight_layout()
plt.savefig(os.path.join(results_plots, "publication_roc_pr_curves.png"), dpi=400)
plt.close()

print("\nPART A FINAL STATISTICAL HARDENING COMPLETE")
