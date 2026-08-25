
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Add project root to sys.path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, base_dir)

# Import Existing Modules
from src.quantum.pqkr import PQKR
from src.quantum.readout import QuantumReadoutSVM
from src.models.transformer import UnsupervisedTemporalTransformer
from src.physics.latent_extractor import extract_mrdmd_features, FrozenDCNEncoder
from src.physics.neural_ode import ContinuousNeuralODE
from src.physics.physics_loss import get_pinn_anomaly_score
from src.fusion.learned_fuser import LearnedFusionNetwork
from src.fusion.trigger import PhaseTransitionTrigger

# Define Output Directory per Prompt
output_path = os.path.join(base_dir, 'results', 'final_pipeline')
os.makedirs(output_path, exist_ok=True)

def execute_final_architecture():
    """
    Final Architecture Implementation for Q1 Journal Validation.
    Fuses Quantum-SVM, Temporal-Transformer, and PINN-ODE into a single SI result.
    """
    data_dir = os.path.join(base_dir, 'data', 'processed')
    healthy_lib = np.load(os.path.join(data_dir, "healthy_windows.npy"))[:100]
    fault_lib = np.load(os.path.join(data_dir, "fault_windows.npy"))[:100]
    
    print("Step 1: Koopman Spectral Feature Extraction...")
    feat_H = np.array([extract_mrdmd_features(w) for w in healthy_lib])
    feat_F = np.array([extract_mrdmd_features(w) for w in fault_lib])
    
    scaler = StandardScaler()
    feat_H_n = scaler.fit_transform(feat_H)
    feat_F_n = scaler.transform(feat_F)
    pca = PCA(n_components=5, random_state=42)
    feat_H_pca = pca.fit_transform(feat_H_n)
    feat_F_pca = pca.transform(feat_F_n)
    
    print("Step 2: PQKR Feature Extraction...")
    pqkr = PQKR(n_qubits=5, seed=42)
    def lift(p):
        v = []
        for x in p:
            s = pqkr.get_state(x)
            v.append(np.concatenate([np.real(s), np.imag(s)]))
        return np.array(v)
    q_H = lift(feat_H_pca)
    q_F = lift(feat_F_pca)
    
    # UPGRADE 1: One-Class SVM Readout
    print("Step 3: One-Class SVM Quantum Readout Layer...")
    K_HH = pqkr.compute_kernel(feat_H_pca, feat_H_pca)
    K_FH = pqkr.compute_kernel(feat_F_pca, feat_H_pca)
    
    svm_readout = QuantumReadoutSVM(kernel='precomputed')
    svm_readout.fit(K_HH)
    q_div_score_H = svm_readout.score(K_HH)
    q_div_score_F = svm_readout.score(K_FH)
    
    # Save Quantum Divergence Scores
    all_q_div = np.concatenate([q_div_score_H, q_div_score_F])
    pd.DataFrame(all_q_div, columns=["QuantumDivergenceScore"]).to_csv(
        os.path.join(output_path, "quantum_divergence_scores.csv"), index=False
    )
    
    print("Step 4: Temporal Transformer Encoding...")
    # seq_len=10 windowing
    def seq(d):
        return torch.tensor([d[i:i+10] for i in range(len(d)-9)], dtype=torch.float32)
    Q_H_s = seq(q_H)
    Q_F_s = seq(q_F)
    
    trans = UnsupervisedTemporalTransformer(input_dim=64)
    r_H, _ = trans(Q_H_s)
    r_F, _ = trans(Q_F_s)
    t_err_H = torch.mean((Q_H_s - r_H)**2, dim=(1,2)).detach().numpy()
    t_err_F = torch.mean((Q_F_s - r_F)**2, dim=(1,2)).detach().numpy()
    
    print("Step 5: Physics-Guided Neural ODE...")
    dcn = FrozenDCNEncoder(input_dim=64)
    p_err_H = get_pinn_anomaly_score(ContinuousNeuralODE(8), dcn(torch.tensor(q_H, dtype=torch.float32)))
    p_err_F = get_pinn_anomaly_score(ContinuousNeuralODE(8), dcn(torch.tensor(q_F, dtype=torch.float32)))
    
    print("Step 6: Learned Fusion (SI Production)...")
    m = min(len(q_div_score_H), len(t_err_H), len(p_err_H))
    def fuse_m(q, t, p, k):
        return np.column_stack([q[-m:], t[-m:], p[-m:], k[-m:, 0]])
    
    in_H = fuse_m(q_div_score_H, t_err_H, p_err_H, feat_H_n)
    in_F = fuse_m(q_div_score_F, t_err_F, p_err_F, feat_F_n)
    
    fuser = LearnedFusionNetwork(4)
    fuser.train_fusion(np.concatenate([in_H, in_F]), np.concatenate([np.zeros(m), np.ones(m)]))
    
    with torch.no_grad():
        SI_H = fuser(torch.tensor(in_H, dtype=torch.float32)).numpy().ravel()
        SI_F = fuser(torch.tensor(in_F, dtype=torch.float32)).numpy().ravel()
        SI_all = np.concatenate([SI_H, SI_F])
        
    pd.DataFrame(SI_all, columns=["SI"]).to_csv(os.path.join(output_path, "si_timeseries.csv"), index=False)
    
    # UPGRADE 2: Phase Transition Detection
    print("Step 7: Isolation Forest Phase Transition Detection...")
    trigger = PhaseTransitionTrigger(window_size=10)
    death_idx = trigger.detect_transition(SI_all)
    
    with open(os.path.join(output_path, "phase_transition_index.txt"), "w") as f:
        f.write(str(death_idx))
    
    # Visualizations
    plt.figure(figsize=(10, 5))
    plt.plot(SI_all, label="Instability Score (SI)")
    plt.axvline(x=m, color='k', linestyle=':', label="Fault Birth")
    if death_idx:
        plt.axvline(x=death_idx, color='r', linestyle='--', label=f"Trigger (Idx:{death_idx})")
    plt.title("FINAL SYSTEM: PHASE TRANSITION TRIGGER")
    plt.legend()
    plt.savefig(os.path.join(output_path, "transition_plot.png"))
    plt.close()
    
    # ROC/PR for validation
    y_true = np.concatenate([np.zeros(m), np.ones(m)])
    fpr, tpr, _ = roc_curve(y_true, SI_all)
    pre, rec, _ = precision_recall_curve(y_true, SI_all)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.4f}")
    plt.title("Final System ROC")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(rec, pre, label=f"PR-AUC={auc(rec, pre):.4f}")
    plt.title("Final System PR Curve")
    plt.legend()
    plt.savefig(os.path.join(output_path, "roc_pr_curves.png"))
    plt.close()
    
    print("\nFINAL ARCHITECTURE IMPLEMENTED — READY FOR VALIDATION")

if __name__ == "__main__":
    execute_final_architecture()
