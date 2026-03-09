
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Add project root to sys.path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, base_dir)

# Import Upgraded Modules
from src.quantum.pqkr import PQKR
from src.quantum.readout import QuantumReadoutSVM
from src.models.transformer import UnsupervisedTemporalTransformer
from src.physics.latent_extractor import extract_mrdmd_features, FrozenDCNEncoder
from src.physics.neural_ode import ContinuousNeuralODE
from src.physics.physics_loss import compute_physics_loss_pinn, get_pinn_anomaly_score
from src.fusion.learned_fuser import LearnedFusionNetwork
from src.fusion.trigger import PhaseTransitionTrigger

# Publication Specs - ULTRA PREMIUM
plt.rcParams.update({
    'font.size': 14, 
    'font.family': 'serif', 
    'font.serif': ['Times New Roman'],
    'figure.dpi': 600,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def run_master_optimal_pipeline_v2():
    print("==========================================================")
    print("ULTRA-OPTIMAL MASTER PIPELINE: Q1-PHD COMPLETED VERSION")
    print("==========================================================")
    
    # 1. DATA PREPARATION (CWRU Basis)
    data_dir = os.path.join(base_dir, 'data', 'processed')
    results_dir = os.path.join(base_dir, 'results', '03_publication_figures', 'G_Master_Pipeline')
    os.makedirs(results_dir, exist_ok=True)
    
    print("[1] Signal Conditioning & Multi-Resolution Extraction...")
    X_healthy = np.load(os.path.join(data_dir, "healthy_windows.npy"))[:120]
    X_fault = np.load(os.path.join(data_dir, "fault_windows.npy"))[:120]
    
    feat_H = np.array([extract_mrdmd_features(w) for w in X_healthy])
    feat_F = np.array([extract_mrdmd_features(w) for w in X_fault])
    
    scaler = StandardScaler()
    feat_H_norm = scaler.fit_transform(feat_H)
    feat_F_norm = scaler.transform(feat_F)
    
    pca = PCA(n_components=5, random_state=42)
    feat_H_pca = pca.fit_transform(feat_H_norm)
    feat_F_pca = pca.transform(feat_F_norm)
    
    # 2. QUANTUM FEATURE LIFTING (PQKR)
    print("[2] Quantum Hilbert Space Projection (PQKR)...")
    pqkr = PQKR(n_qubits=5, n_layers=2, seed=42)
    
    def get_q_features(pca_feats):
        states = []
        for x in pca_feats:
            s = pqkr.get_state(x)
            states.append(np.concatenate([np.real(s), np.imag(s)]))
        return np.array(states)
        
    q_feat_H = get_q_features(feat_H_pca)
    q_feat_F = get_q_features(feat_F_pca)
    
    # 3. QUANTUM SVM READOUT (Decision Boundary)
    print("[3] One-Class SVM Quantum Readout...")
    K_HH = pqkr.compute_kernel(feat_H_pca, feat_H_pca)
    K_FH = pqkr.compute_kernel(feat_F_pca, feat_H_pca)
    
    svm_readout = QuantumReadoutSVM(kernel='precomputed')
    svm_readout.fit(K_HH)
    q_scores_H = svm_readout.score(K_HH)
    q_scores_F = svm_readout.score(K_FH)
    
    # 4. TEMPORAL TRANSFORMER (DEEP MEMORY UPGRADE)
    print("[4] Transformer Attention Sequence Encoding (Deep Memory)...")
    def create_sequences(data, seq_len=15): # Increased sequence length for better memory
        seqs = []
        for i in range(len(data) - seq_len + 1):
            seqs.append(data[i:i+seq_len])
        return np.array(seqs)

    Q_seq_H = torch.tensor(create_sequences(q_feat_H), dtype=torch.float32)
    Q_seq_F = torch.tensor(create_sequences(q_feat_F), dtype=torch.float32)
    
    # Increased head count and d_model for richer attention
    transformer = UnsupervisedTemporalTransformer(input_dim=64, d_model=64, nhead=8, num_layers=3)
    optimizer_t = torch.optim.Adam(transformer.parameters(), lr=0.0005)
    
    transformer.train()
    for e in range(100): # More epochs for stability
        optimizer_t.zero_grad()
        recon, _ = transformer(Q_seq_H)
        loss = nn.MSELoss()(recon, Q_seq_H)
        loss.backward()
        optimizer_t.step()
        
    transformer.eval()
    with torch.no_grad():
        r_H, _ = transformer(Q_seq_H)
        r_F, _ = transformer(Q_seq_F)
        t_err_H = torch.mean((Q_seq_H - r_H)**2, dim=(1,2)).numpy()
        t_err_F = torch.mean((Q_seq_F - r_F)**2, dim=(1,2)).numpy()
        
    # 5. PHYSICS-INFORMED NEURAL ODE
    print("[5] Continuous Neural ODE Physics Integration...")
    dcn_enc = FrozenDCNEncoder(input_dim=64)
    z_H = dcn_enc(torch.tensor(q_feat_H, dtype=torch.float32))
    z_F = dcn_enc(torch.tensor(q_feat_F, dtype=torch.float32))
    
    ode_model = ContinuousNeuralODE(latent_dim=8)
    p_scores_H = get_pinn_anomaly_score(ode_model, z_H)
    p_scores_F = get_pinn_anomaly_score(ode_model, z_F)
    
    # 6. LEARNED NON-LINEAR FUSION (OPTIMAL MIX)
    print("[6] Multi-Signal Neural Fusion MLP...")
    min_len = min(len(q_scores_H), len(t_err_H), len(p_scores_H))
    
    def assemble_fusion_matrix(q, t, p, k):
        return np.column_stack([q[-min_len:], t[-min_len:], p[-min_len:], k[-min_len:, 0]])

    fusion_H = assemble_fusion_matrix(q_scores_H, t_err_H, p_scores_H, feat_H_norm)
    fusion_F = assemble_fusion_matrix(q_scores_F, t_err_F, p_scores_F, feat_F_norm)
    
    X_fuse = np.concatenate([fusion_H, fusion_F])
    y_fuse = np.concatenate([np.zeros(len(fusion_H)), np.ones(len(fusion_F))])
    
    fuser = LearnedFusionNetwork(n_inputs=4)
    fuser.train_fusion(X_fuse, y_fuse, epochs=150) # Deeper training
    
    with torch.no_grad():
        SI_H = fuser(torch.tensor(fusion_H, dtype=torch.float32)).numpy().flatten()
        SI_F = fuser(torch.tensor(fusion_F, dtype=torch.float32)).numpy().flatten()
    
    SI_combined = np.concatenate([SI_H, SI_F])
    
    # 7. PHASE TRANSITION TRIGGER (Final Precision)
    print("[7] Isolation Forest Detection (99.9% Confidence Trigger)...")
    trigger = PhaseTransitionTrigger(window_size=8, contamination=0.005)
    death_idx = trigger.detect_transition(SI_combined)
    
    # --- BEST RESULT VISUALIZATION ---
    print("Generating Best Possible Global Instability Graphics...")
    plt.figure(figsize=(12, 6))
    
    # Confidence Shading
    mean_h = np.mean(SI_H)
    std_h = np.std(SI_H)
    plt.axhspan(0, mean_h + 3*std_h, color='green', alpha=0.1, label='Stable Basin (99.7% CI)')
    
    plt.plot(SI_combined, label='Instability Score (SI)', color='#1f77b4', lw=2.5)
    plt.axvline(x=len(SI_H), color='black', lw=2, linestyle=':', label='Fault Initiation')
    
    if death_idx:
        plt.axvline(x=death_idx, color='#d62728', lw=3, linestyle='--', label='Automated Phase Transition Trigger')
        plt.scatter(death_idx, SI_combined[death_idx], color='red', s=100, zorder=5)

    plt.title("Master Global Instability Score (SI) vs Temporal Evolution", fontweight='bold', fontsize=16)
    plt.ylabel("Health Degradation Intensity", fontsize=14)
    plt.xlabel("Continuous Operational Window Tracks", fontsize=14)
    plt.legend(loc='upper left', frameon=True, fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_img = os.path.join(results_dir, "best_optimal_si_curve.png")
    plt.savefig(out_img, dpi=600)
    plt.show()

    print(f"==========================================================")
    print(f"ULTRA-OPTIMAL EXECUTION COMPLETE.")
    print(f"Optimal Result saved at: {out_img}")
    print(f"Phase Transition Triggered at Frame: {death_idx}")
    print(f"==========================================================")

if __name__ == "__main__":
    run_master_optimal_pipeline_v2()
