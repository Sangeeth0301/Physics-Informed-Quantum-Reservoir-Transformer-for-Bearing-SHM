import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import ttest_ind, ks_2samp
from sklearn.metrics.pairwise import rbf_kernel

# Set publication quality parameters
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 300, 'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'axes.titleweight': 'bold',
    'axes.prop_cycle': plt.cycler('color', ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'])
})

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, base_dir)

from pydmd import MrDMD, DMD
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.quantum.pqkr import PQKR
from src.quantum.metrics import compute_mmd, frobenius_divergence
from src.data_prep.signal_processing import preprocess_bearing_signal

data_dir = os.path.join(base_dir, 'data', 'processed')
results_plots = os.path.join(base_dir, 'results', 'plots')
results_tables = os.path.join(base_dir, 'results', 'tables')
os.makedirs(results_plots, exist_ok=True)
os.makedirs(results_tables, exist_ok=True)

print("PHASE 4.5: SYSTEM-LEVEL VALIDATION & ABLATION")

# -------------------------------------------------------------
# 0. Load Data and Extract Base Features
# -------------------------------------------------------------
print("Loading CWRU and IMS data...")
healthy_windows_full = np.load(os.path.join(data_dir, "healthy_windows.npy"))
fault_windows_full = np.load(os.path.join(data_dir, "fault_windows.npy"))
healthy_windows = healthy_windows_full[:40]
fault_windows = fault_windows_full[:40]

try:
    ims_windows = np.load(os.path.join(data_dir, "ims_stride_windows.npy"))
except FileNotFoundError:
    ims_windows = None

def extract_mrdmd_features(signal, delay=60, svd_rank=12):
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

print("Extracting CWRU features...")
f_H_base = np.array([extract_mrdmd_features(w) for w in healthy_windows])
f_F_base = np.array([extract_mrdmd_features(w) for w in fault_windows])

scaler = StandardScaler()
X_H_norm = scaler.fit_transform(f_H_base)
X_F_norm = scaler.transform(f_F_base)

n_qubits = 5
pca = PCA(n_components=n_qubits, random_state=42)
X_H_pca = pca.fit_transform(X_H_norm)
X_F_pca = pca.transform(X_F_norm)

# -------------------------------------------------------------
# DCN Autoencoder Definition (No Physics ODE)
# -------------------------------------------------------------
class DynamicalConsistencyNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ELU(),
            nn.Linear(32, 16), nn.ELU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ELU(),
            nn.Linear(16, 32), nn.ELU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

def train_dcn(train_data, epochs=150, seed=42):
    torch.manual_seed(seed)
    model = DynamicalConsistencyNetwork(train_data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        x_rec = model(train_data)
        loss = criterion(x_rec, train_data)
        loss.backward()
        optimizer.step()
    model.eval()
    return model

def get_quantum_states(pqkr, X):
    return np.array([np.concatenate([np.real(pqkr.get_state(x)), np.imag(pqkr.get_state(x))]) for x in X])

def get_separation_metrics(K_HH, K_FF, K_HF):
    intra_H = np.mean(K_HH)
    intra_F = np.mean(K_FF)
    inter = np.mean(K_HF)
    intra = (intra_H + intra_F) / 2.0
    frob = frobenius_divergence(K_HH, K_FF)
    mmd = compute_mmd(K_HH, K_FF, K_HF)
    sep_ratio = intra / (inter + 1e-8)
    return frob, mmd, intra, inter, sep_ratio

# -------------------------------------------------------------
# STEP 1: Component Ablation Study
# -------------------------------------------------------------
print("==> STEP 1: Component Ablation Study")
ablation_res = []

# Model A: mrDMD only
gamma = 1.0 / n_qubits
K_HH_A = rbf_kernel(X_H_pca, X_H_pca, gamma=gamma)
K_FF_A = rbf_kernel(X_F_pca, X_F_pca, gamma=gamma)
K_HF_A = rbf_kernel(X_H_pca, X_F_pca, gamma=gamma)
frob_A, mmd_A, intra_A, inter_A, sep_A = get_separation_metrics(K_HH_A, K_FF_A, K_HF_A)
ablation_res.append({"Model ID": "A", "Components": "mrDMD only", "Frob": frob_A, "MMD": mmd_A, "Intra": intra_A, "Inter": inter_A, "Sep_Ratio": sep_A})

# Model B: mrDMD + PQKR
pqkr = PQKR(n_qubits=n_qubits, n_layers=2, seed=42)
np.random.seed(42)  # Noise injection for stability
X_H_noisy = X_H_pca + np.random.normal(0, 0.02, X_H_pca.shape)
X_F_noisy = X_F_pca + np.random.normal(0, 0.02, X_F_pca.shape)

K_HH_B = pqkr.compute_kernel(X_H_noisy, X_H_noisy)
K_FF_B = pqkr.compute_kernel(X_F_noisy, X_F_noisy)
K_HF_B = pqkr.compute_kernel(X_H_noisy, X_F_noisy)
frob_B, mmd_B, intra_B, inter_B, sep_B = get_separation_metrics(K_HH_B, K_FF_B, K_HF_B)
ablation_res.append({"Model ID": "B", "Components": "mrDMD + PQKR", "Frob": frob_B, "MMD": mmd_B, "Intra": intra_B, "Inter": inter_B, "Sep_Ratio": sep_B})

# Model C: mrDMD + PQKR + DCN
train_data_C = torch.tensor(get_quantum_states(pqkr, X_H_noisy), dtype=torch.float32)
test_fault_C = torch.tensor(get_quantum_states(pqkr, X_F_noisy), dtype=torch.float32)

model_dcnC = train_dcn(train_data_C, seed=42)
with torch.no_grad():
    err_H = torch.mean((model_dcnC(train_data_C) - train_data_C)**2, dim=1).numpy()
    err_F = torch.mean((model_dcnC(test_fault_C) - test_fault_C)**2, dim=1).numpy()

# Pseudo-kernel matrix for distance metrics
K_HH_C = np.exp(-np.abs(err_H[:, None] - err_H[None, :]))
K_FF_C = np.exp(-np.abs(err_F[:, None] - err_F[None, :]))
K_HF_C = np.exp(-np.abs(err_H[:, None] - err_F[None, :]))
frob_C, mmd_C, intra_C, inter_C, sep_C = get_separation_metrics(K_HH_C, K_FF_C, K_HF_C)
ablation_res.append({"Model ID": "C", "Components": "mrDMD + PQKR + DCN", "Frob": frob_C, "MMD": mmd_C, "Intra": intra_C, "Inter": inter_C, "Sep_Ratio": sep_C})

# Model D: mrDMD + PQKR + DCN + SI fusion
mu_recon, std_recon = np.mean(err_H), np.std(err_H)
def get_si(err, mu, std):
    z = (err - mu) / (std + 1e-8)
    return 1 / (1 + np.exp(-(z - 3.0)))

si_H = get_si(err_H, mu_recon, std_recon)
si_F = get_si(err_F, mu_recon, std_recon)
K_HH_D = np.exp(-np.abs(si_H[:, None] - si_H[None, :]))
K_FF_D = np.exp(-np.abs(si_F[:, None] - si_F[None, :]))
K_HF_D = np.exp(-np.abs(si_H[:, None] - si_F[None, :]))
frob_D, mmd_D, intra_D, inter_D, sep_D = get_separation_metrics(K_HH_D, K_FF_D, K_HF_D)
ablation_res.append({"Model ID": "D", "Components": "mrDMD + PQKR + DCN + SI", "Frob": frob_D, "MMD": mmd_D, "Intra": intra_D, "Inter": inter_D, "Sep_Ratio": sep_D})

df_abl = pd.DataFrame(ablation_res)
df_abl.to_csv(os.path.join(results_tables, "component_ablation.csv"), index=False)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_abl))
width = 0.35
ax.bar(x - width/2, df_abl["MMD"], width, label='MMD (Discrepancy)', color='#0072B2')
ax.bar(x + width/2, df_abl["Sep_Ratio"], width, label='Separation Ratio (Intra / Inter)', color='#D55E00')
ax.set_xticks(x)
ax.set_xticklabels(df_abl["Model ID"])
ax.set_ylabel("Metric Magnitude")
ax.set_yscale('log')
ax.set_title("Component Ablation: Stepwise Separation Enhancements")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_plots, "component_ablation_bar.png"))
plt.close()

# -------------------------------------------------------------
# STEP 2: Multi-Seed Statistical Hardening
# -------------------------------------------------------------
print("==> STEP 2: Multi-Seed Statistical Hardening")
seed_metrics = []
seeds = list(range(100, 110))

for s in seeds:
    pqkr_s = PQKR(n_qubits=n_qubits, n_layers=2, seed=s)
    np.random.seed(s)
    XH_s = X_H_pca + np.random.normal(0, 0.02, X_H_pca.shape)
    XF_s = X_F_pca + np.random.normal(0, 0.02, X_F_pca.shape)
    
    tr_H_s = torch.tensor(get_quantum_states(pqkr_s, XH_s), dtype=torch.float32)
    te_F_s = torch.tensor(get_quantum_states(pqkr_s, XF_s), dtype=torch.float32)
    
    m_dcn_s = train_dcn(tr_H_s, epochs=100, seed=s)
    with torch.no_grad():
        e_H_s = torch.mean((m_dcn_s(tr_H_s) - tr_H_s)**2, dim=1).numpy()
        e_F_s = torch.mean((m_dcn_s(te_F_s) - te_F_s)**2, dim=1).numpy()
        
    m_e_s, std_e_s = np.mean(e_H_s), np.std(e_H_s)
    si_H_s = get_si(e_H_s, m_e_s, std_e_s)
    si_F_s = get_si(e_F_s, m_e_s, std_e_s)
    
    KHH = pqkr_s.compute_kernel(XH_s, XH_s)
    KFF = pqkr_s.compute_kernel(XF_s, XF_s)
    KHF = pqkr_s.compute_kernel(XH_s, XF_s)
    
    seed_metrics.append({
        "Seed": s,
        "Mean_SI_H": np.mean(si_H_s),
        "Mean_SI_F": np.mean(si_F_s),
        "MMD_Quantum": compute_mmd(KHH, KFF, KHF),
        "DCN_MSE_H": m_e_s,
        "DCN_MSE_F": np.mean(e_F_s)
    })

df_seeds = pd.DataFrame(seed_metrics)
summary = []
for col in df_seeds.columns[1:]:
    m_val = df_seeds[col].mean()
    s_val = df_seeds[col].std()
    ci = 1.96 * s_val / np.sqrt(len(seeds))
    summary.append({
        "Metric": col,
        "Mean": m_val,
        "Std": s_val,
        "95% CI ±": ci
    })
pd.DataFrame(summary).to_csv(os.path.join(results_tables, "statistical_summary.csv"), index=False)

# -------------------------------------------------------------
# STEP 3: IMS Early Warning Curve
# -------------------------------------------------------------
print("==> STEP 3: IMS Early Warning Curve")
if ims_windows is not None:
    ims_feats = np.array([extract_mrdmd_features(w) for w in ims_windows])
    ims_pca_norm = pca.transform(scaler.transform(ims_feats))
    ims_st = torch.tensor(get_quantum_states(pqkr, ims_pca_norm), dtype=torch.float32)
    
    with torch.no_grad():
        ims_err = torch.mean((model_dcnC(ims_st) - ims_st)**2, dim=1).numpy()
    
    ims_si = get_si(ims_err, mu_recon, std_recon)
    days = np.linspace(0, 35, len(ims_si))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, ims_si, color='#0072B2', linewidth=2.5, label='Instability Score (SI)')
    ax.fill_between(days, 0, 0.2, color='#009E73', alpha=0.3, label='Healthy Baseline Bound')
    ax.axvline(21, color='#D55E00', linestyle='--', linewidth=2, label='Incipient Fault Induced')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=2, label='Alarm Threshold $\\tau=0.5$')
    
    ax.set_title("IMS Dataset: SI Early Warning Progression")
    ax.set_xlabel("Operational Time (Days)")
    ax.set_ylabel("Instability Score (SI) $\in [0, 1]$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_plots, "ims_early_warning_curve.png"))
    plt.close()
else:
    print("IMS data not found.")

# -------------------------------------------------------------
# STEP 4: Noise Robustness Experiment
# -------------------------------------------------------------
print("==> STEP 4: Noise Robustness Experiment")
snr_db_levels = [None, 20, 10, 5]
rob_res = []

def add_noise(signal, snr_db):
    if snr_db is None: return signal
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

print("Processing Noise Robustness limits... (this takes a moment)")
for snr in snr_db_levels:
    f_H_n = np.array([extract_mrdmd_features(add_noise(w, snr)) for w in healthy_windows])
    f_F_n = np.array([extract_mrdmd_features(add_noise(w, snr)) for w in fault_windows])
    
    xh_n = pca.transform(scaler.transform(f_H_n))
    xf_n = pca.transform(scaler.transform(f_F_n))
    
    kh_n = pqkr.compute_kernel(xh_n, xh_n)
    kf_n = pqkr.compute_kernel(xf_n, xf_n)
    khf_n = pqkr.compute_kernel(xh_n, xf_n)
    
    r_frob = frobenius_divergence(kh_n, kf_n)
    r_mmd = compute_mmd(kh_n, kf_n, khf_n)
    
    tr_h = torch.tensor(get_quantum_states(pqkr, xh_n), dtype=torch.float32)
    tr_f = torch.tensor(get_quantum_states(pqkr, xf_n), dtype=torch.float32)
    
    with torch.no_grad():
        eh = torch.mean((model_dcnC(tr_h) - tr_h)**2, dim=1).numpy()
        ef = torch.mean((model_dcnC(tr_f) - tr_f)**2, dim=1).numpy()
        
    si_h = get_si(eh, mu_recon, std_recon)
    si_f = get_si(ef, mu_recon, std_recon)
    
    rob_res.append({
        "SNR (dB)": "Clean" if snr is None else str(snr),
        "Frob": r_frob,
        "MMD": r_mmd,
        "SI_Separation": np.mean(si_f) - np.mean(si_h)
    })

df_rob = pd.DataFrame(rob_res)
df_rob.to_csv(os.path.join(results_tables, "noise_robustness.csv"), index=False)

fig, ax1 = plt.subplots(figsize=(8, 5))
x_labels = df_rob["SNR (dB)"].astype(str)
ax1.plot(x_labels, df_rob["MMD"], marker='o', label='MMD (Quantum Separation)', color='#0072B2', linewidth=2)
ax1.plot(x_labels, df_rob["SI_Separation"], marker='s', label='SI Separation Δ', color='#D55E00', linewidth=2)
ax1.set_xlabel("Signal-to-Noise Ratio (dB) Added to Raw Vibration")
ax1.set_ylabel("Metric Magnitude")
ax1.set_title("Pipeline Robustness Against Induced Additive Gaussian Noise")
ax1.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_plots, "noise_robustness.png"))
plt.close()

# -------------------------------------------------------------
# STEP 5: Statistical Significance Tests
# -------------------------------------------------------------
print("==> STEP 5: Statistical Significance Tests (p-values)")
stat_tests = []
t_stat, p_t = ttest_ind(si_H, si_F, equal_var=False)
stat_tests.append({
    "Test": "Welch's t-test", 
    "Metric": "Final SI Score", 
    "Healthy/Fault Separation": "CWRU DCN Output",
    "p-value": f"{p_t:.3e}", 
    "Significant (p<0.01)": p_t < 0.01})

ks_stat, p_ks = ks_2samp(f_H_base[:, 0], f_F_base[:, 0])
stat_tests.append({
    "Test": "Kolmogorov-Smirnov", 
    "Metric": "mrDMD Spectral Radius", 
    "Healthy/Fault Separation": "CWRU Phase 2 Output",
    "p-value": f"{p_ks:.3e}", 
    "Significant (p<0.01)": p_ks < 0.01})

pd.DataFrame(stat_tests).to_csv(os.path.join(results_tables, "statistical_tests.csv"), index=False)

# -------------------------------------------------------------
# STEP 6: Decision-Level Metrics (ROC, PR, Thresholding)
# -------------------------------------------------------------
print("==> STEP 6: Decision-Level Metrics (ROC, PR, Thresholding)")
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# True labels: Healthy=0, Fault=1
y_true = np.concatenate([np.zeros(len(si_H)), np.ones(len(si_F))])

# 1. DCN Mean Squared Error
y_score_mse = np.concatenate([err_H, err_F])
fpr_mse, tpr_mse, _ = roc_curve(y_true, y_score_mse)
roc_auc_mse = auc(fpr_mse, tpr_mse)
precision_mse, recall_mse, _ = precision_recall_curve(y_true, y_score_mse)
pr_auc_mse = average_precision_score(y_true, y_score_mse)

# 2. Instability Score (SI)
y_score_si = np.concatenate([si_H, si_F])
fpr_si, tpr_si, _ = roc_curve(y_true, y_score_si)
roc_auc_si = auc(fpr_si, tpr_si)
precision_si, recall_si, _ = precision_recall_curve(y_true, y_score_si)
pr_auc_si = average_precision_score(y_true, y_score_si)

# Save Metrics DataFrame
dec_metrics = [
    {"Metric": "ROC-AUC", "DCN_MSE": roc_auc_mse, "SI_Score": roc_auc_si},
    {"Metric": "PR-AUC", "DCN_MSE": pr_auc_mse, "SI_Score": pr_auc_si}
]
pd.DataFrame(dec_metrics).to_csv(os.path.join(results_tables, "decision_metrics.csv"), index=False)

# Plotting ROC and PR
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# ROC
axs[0].plot(fpr_mse, tpr_mse, label=f'DCN MSE (AUC = {roc_auc_mse:.4f})', color='#D55E00', linestyle='--')
axs[0].plot(fpr_si, tpr_si, label=f'SI Score (AUC = {roc_auc_si:.4f})', color='#0072B2', linewidth=2)
axs[0].plot([0, 1], [0, 1], 'k:')
axs[0].set_title("ROC Curve")
axs[0].set_xlabel("False Positive Rate")
axs[0].set_ylabel("True Positive Rate")
axs[0].legend(loc="lower right")

# PR
axs[1].plot(recall_mse, precision_mse, label=f'DCN MSE (PR-AUC = {pr_auc_mse:.4f})', color='#D55E00', linestyle='--')
axs[1].plot(recall_si, precision_si, label=f'SI Score (PR-AUC = {pr_auc_si:.4f})', color='#0072B2', linewidth=2)
axs[1].set_title("Precision-Recall Curve")
axs[1].set_xlabel("Recall")
axs[1].set_ylabel("Precision")
axs[1].legend(loc="lower left")

plt.tight_layout()
plt.savefig(os.path.join(results_plots, "roc_pr_curves.png"))
plt.close()

print("\nPHASE 4.5 VALIDATION COMPLETE — READY FOR REVIEW")
