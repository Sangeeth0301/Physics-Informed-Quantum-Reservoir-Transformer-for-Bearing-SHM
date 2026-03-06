import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, base_dir)

from src.quantum.pqkr import PQKR
from src.physics.latent_extractor import FrozenDCNEncoder, extract_latent_trajectory, extract_mrdmd_features
from src.physics.neural_ode import DiscreteNeuralODE
from src.physics.physics_loss import compute_physics_residual, compute_physics_loss

data_dir = os.path.join(base_dir, 'data', 'processed')
results_plots = os.path.join(base_dir, 'results', 'physics')
os.makedirs(results_plots, exist_ok=True)

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.dpi': 300, 'font.family': 'serif', 'font.serif': ['Times New Roman']
})

print("PHASE 8: PHYSICS-GUIDED LATENT EVOLUTION (NEURAL ODE)")

# --- 1. Latent Extraction ---
print("==> 1. Latent Extraction")
torch.manual_seed(42)
np.random.seed(42)

healthy_windows = np.load(os.path.join(data_dir, "healthy_windows.npy"))[:40]
fault_windows = np.load(os.path.join(data_dir, "fault_windows.npy"))[:40]

f_H_base = np.array([extract_mrdmd_features(w) for w in healthy_windows])
f_F_base = np.array([extract_mrdmd_features(w) for w in fault_windows])

scaler = StandardScaler()
X_H_norm = scaler.fit_transform(f_H_base)
X_F_norm = scaler.transform(f_F_base)

pca = PCA(n_components=5, random_state=42)
X_H_pca = pca.fit_transform(X_H_norm)
X_F_pca = pca.transform(X_F_norm)

pqkr = PQKR(n_qubits=5, n_layers=2, seed=42)
dcn_enc = FrozenDCNEncoder(input_dim=64) # 32 complex -> 64 real

# dummy initialized DCN just to extract deterministic latents for phase 8
dcn_enc.eval() 

Z_H = extract_latent_trajectory(healthy_windows, scaler, pca, dcn_enc, pqkr)
Z_F = extract_latent_trajectory(fault_windows, scaler, pca, dcn_enc, pqkr)

Z_H = torch.tensor(Z_H, dtype=torch.float32)
Z_F = torch.tensor(Z_F, dtype=torch.float32)

# --- 2. Latent Trajectory Construction ---
print("==> 2. Latent Trajectory Construction")
def create_pairs(Z):
    Z_t = Z[:-1]
    Z_t_plus_1 = Z[1:]
    return Z_t, Z_t_plus_1

Z_H_t, Z_H_t_plus_1 = create_pairs(Z_H)
Z_F_t, Z_F_t_plus_1 = create_pairs(Z_F)

# --- 3. Neural ODE Model ---
print("==> 3. Neural ODE Model Initialization")
model = DiscreteNeuralODE(latent_dim=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- 4, 5, 6. Training Objective & Prediction ---
print("==> 4-6. Training Neural ODE with Physics Constraint")
epochs = 200
dt = 1.0 # time step between windows
lambda_phys = 0.05
c_damp = 0.1
k_stiff = 0.05

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Predict step: z_pred = z + dzdt * dt
    dzdt = model(Z_H_t)
    Z_pred = Z_H_t + dzdt * dt
    
    # L_dyn
    L_dyn = torch.mean((Z_H_t_plus_1 - Z_pred)**2)
    
    # We construct the full temporal sequence of predictions to apply physics
    # Sequence: [z_0, z_pred_1, z_pred_2, ...]
    seq_pred = torch.cat([Z_H[0:1], Z_pred], dim=0)
    
    # L_phys
    r_phys = compute_physics_residual(seq_pred, dt=dt, c=c_damp, k=k_stiff)
    L_phys = compute_physics_loss(r_phys)
    
    L_total = L_dyn + lambda_phys * L_phys
    L_total.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} | L_dyn: {L_dyn.item():.6f} | L_phys: {L_phys.item():.6f} | L_total: {L_total.item():.6f}")

# --- 7. Physics Anomaly Score ---
print("==> 7. Physics Anomaly Score Computation")
model.eval()

with torch.no_grad():
    dzdt_H = model(Z_H_t)
    Z_pred_H = Z_H_t + dzdt_H * dt
    seq_H = torch.cat([Z_H[0:1], Z_pred_H], dim=0)
    r_phys_H = compute_physics_residual(seq_H, dt=dt, c=c_damp, k=k_stiff)
    # physics anomaly score for each window: ||r_phys||
    # Notice r_phys is length T-2
    score_H = torch.abs(r_phys_H).numpy()
    
    dzdt_F = model(Z_F_t)
    Z_pred_F = Z_F_t + dzdt_F * dt
    seq_F = torch.cat([Z_F[0:1], Z_pred_F], dim=0)
    r_phys_F = compute_physics_residual(seq_F, dt=dt, c=c_damp, k=k_stiff)
    score_F = torch.abs(r_phys_F).numpy()

np.save(os.path.join(results_plots, "physics_scores_H.npy"), score_H)
np.save(os.path.join(results_plots, "physics_scores_F.npy"), score_F)

# --- 8. Visualization ---
print("==> 8. Visualization")

# Plot 1: Latent trajectory evolution (healthy vs fault)
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(seq_H[:, 0].numpy(), label='Healthy $z_1$ (Predicted)', color='#009E73', marker='o')
ax1.plot(seq_F[:, 0].numpy(), label='Fault $z_1$ (Predicted)', color='#D55E00', marker='x')
ax1.set_title("Latent Trajectory Evolution")
ax1.set_xlabel("Time Step (Window)")
ax1.set_ylabel("Latent State $z_1$")
ax1.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_plots, "latent_trajectory_evolution.png"))
plt.close()

# Plot 2: Physics residual histogram (healthy vs fault)
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.hist(score_H, bins=15, alpha=0.7, label='Healthy Residuals', color='#009E73')
ax2.hist(score_F, bins=15, alpha=0.7, label='Fault Residuals', color='#D55E00')
ax2.set_title("Physics Residual Histogram")
ax2.set_xlabel("Physics Anomaly Score $||r_{phys}||$")
ax2.set_ylabel("Frequency")
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_plots, "physics_residual_histogram.png"))
plt.close()

# Plot 3: Physics residual over time
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(score_H, label='Healthy Residuals', color='#009E73', marker='o')
ax3.plot(score_F, label='Fault Residuals', color='#D55E00', marker='x')
ax3.set_title("Physics Residual Over Time")
ax3.set_xlabel("Time Step (Window)")
ax3.set_ylabel("Physics Anomaly Score $||r_{phys}||$")
ax3.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_plots, "physics_residual_over_time.png"))
plt.close()

print("\nPHASE 8 PHYSICS ODE COMPLETE — READY FOR REVIEW")
