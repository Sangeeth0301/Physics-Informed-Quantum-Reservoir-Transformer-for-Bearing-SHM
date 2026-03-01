import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

# Set publication quality parameters
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.dpi': 300, 'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'axes.titleweight': 'bold'
})

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'data', 'processed')
results_plots = os.path.join(base_dir, 'results', 'plots')
results_tables = os.path.join(base_dir, 'results', 'tables')

sys.path.insert(0, base_dir)
from src.data_prep.signal_processing import preprocess_bearing_signal
from pydmd import MrDMD, DMD
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.quantum.pqkr import PQKR

print("=========================================================================")
print(" Phase 4: Dynamical Consistency Network (Dense DCN) + Physics ODE ")
print("=========================================================================")

print("[1] Structuring CWRU Data for Deep Learning...")
healthy_windows = np.load(os.path.join(data_dir, "healthy_windows.npy"))
fault_windows   = np.load(os.path.join(data_dir, "fault_windows.npy"))

num_windows = 40 # use enough windows to train
healthy_windows = healthy_windows[:num_windows]
fault_windows   = fault_windows[:num_windows]

def extract_features(signal, delay=60, svd_rank=12):
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

f_H = np.array([extract_features(w) for w in healthy_windows])
f_F = np.array([extract_features(w) for w in fault_windows])

scaler = StandardScaler()
X_H_norm = scaler.fit_transform(f_H)
X_F_norm = scaler.transform(f_F)

n_qubits = 5
pca = PCA(n_components=n_qubits, random_state=42)
X_H_pca = pca.fit_transform(X_H_norm)
X_F_pca = pca.transform(X_F_norm)

# Add standard Gaussian spread matching Phase 3
np.random.seed(42)
X_H_noisy = X_H_pca + np.random.normal(0, 0.02, X_H_pca.shape)
X_F_noisy = X_F_pca + np.random.normal(0, 0.02, X_F_pca.shape)

pqkr = PQKR(n_qubits=n_qubits, n_layers=2, seed=42)

print("[2] Projecting CWRU Features explicitly into PQKR Quantum States...")
def get_quantum_states(X):
    states = []
    for x in X:
        state = pqkr.get_state(x)
        state_cat = np.concatenate([np.real(state), np.imag(state)])
        states.append(state_cat)
    return np.array(states)

train_data = torch.tensor(get_quantum_states(X_H_noisy), dtype=torch.float32)
test_healthy = train_data # Testing on seen manifold
test_fault = torch.tensor(get_quantum_states(X_F_noisy), dtype=torch.float32)

input_dim = train_data.shape[1]

print("[3] Defining Dense DCN Autoencoder and Jeffcott/Hertzian Neural ODE...")
class BearingPhysicsODE(nn.Module):
    def __init__(self, c=0.2, k=1.5, alpha=0.5):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(c))
        self.k = nn.Parameter(torch.tensor(k))
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
    def forward(self, t, z):
        # Neural ODE physical projection Constraint: z_1 is x, z_2 is dx/dt
        x = z[:, 0]
        v = z[:, 1]
        
        dx = v
        # Nonlinear restorative force
        restore = self.k * x + self.alpha * (torch.abs(x)**1.5) * torch.sign(x)
        dv = -self.c * v - restore
        
        return torch.stack([dx, dv], dim=1)

class DynamicalConsistencyNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Dense layers natively consume high dimensional quantum states
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Linear(16, 8) # 8-Dimensional Bottleneck manifold
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ELU(),
            nn.Linear(16, 32),
            nn.ELU(),
            nn.Linear(32, input_dim)
        )
        self.physics_ode = BearingPhysicsODE()
        
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        z_phys = z[:, :2] # Enforced Physical Projection
        return x_rec, z_phys

model = DynamicalConsistencyNetwork(input_dim)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

epochs = 150
t_span = torch.tensor([0.0, 0.1]) # Short time slice for physics deviation

print("[4] Training DCN Architecture strictly on Healthy Topology...")
model.train()
for ep in range(epochs):
    optimizer.zero_grad()
    x_rec, z_phys = model(train_data)
    
    # 1. Learn the Healthy Quantum State Geometry
    loss_recon = criterion(x_rec, train_data)
    
    # 2. Learn that evolution must obey Stable Physics Bounds
    z_phys_evolved = odeint(model.physics_ode, z_phys, t_span)[-1]
    loss_phys = criterion(z_phys_evolved, z_phys)
    
    loss = loss_recon + 0.1 * loss_phys
    loss.backward()
    optimizer.step()

print("[5] Generating Deviation Metrics and Fusing Final Instability Score...")
model.eval()

def evaluate(data_tensor):
    with torch.no_grad():
        x_rec, z_phys = model(data_tensor)
        recon_err = torch.mean((x_rec - data_tensor)**2, dim=1).numpy()
        
        z_phys_evolved = odeint(model.physics_ode, z_phys, t_span)[-1]
        phys_err = torch.mean((z_phys_evolved - z_phys)**2, dim=1).numpy()
    return recon_err, phys_err, z_phys.numpy()

h_recon, h_phys, zh = evaluate(test_healthy)
f_recon, f_phys, zf = evaluate(test_fault)

mu_recon, std_recon = np.mean(h_recon), np.std(h_recon)
mu_phys, std_phys = np.mean(h_phys), np.std(h_phys)

def get_si(recon, phys):
    # Mahalanobis Distance Z-Score Fusion Limit
    z_recon = (recon - mu_recon) / (std_recon + 1e-8)
    z_phys  = (phys - mu_phys) / (std_phys + 1e-8)
    combined_z = z_recon + z_phys
    return 1 / (1 + np.exp(- (combined_z - 3.0))) # Boundary constraint bounded

si_h = get_si(h_recon, h_phys)
si_f = get_si(f_recon, f_phys)

# --- MATPLOTLIB Q1 GENERATION ---
print("[6] Exporting Phase 4 Optimal Visualizations...")

# 1. Reconstruction MSE Comparison
fig, ax = plt.subplots(figsize=(8,6))
ax.boxplot([h_recon, f_recon], labels=['Healthy Class', 'Fault Class'], patch_artist=True, 
           boxprops=dict(facecolor="lightblue", color="b"))
ax.set_title("(a) DCN Quantum State Reconstruction Error Isolation")
ax.set_ylabel("Mean Squared Error (MSE)")
plt.savefig(os.path.join(results_plots, "q1_dcn_reconstruction.png"))
plt.close()

# 2. ODE Physical Manifold Representation
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(zh[:,0], zh[:,1], label='Healthy State Bound', alpha=0.9, marker='o', color='green')
ax.scatter(zf[:,0], zf[:,1], label='Fault Constraint Rupture', alpha=0.9, marker='x', color='red')
ax.set_title("(b) Latent Physics ODE Attractor \nProjected into $z_1 \sim x, z_2 \sim \dot{x}$")
ax.set_xlabel("Latent Pseudo-Displacement ($x$)")
ax.set_ylabel("Latent Pseudo-Velocity ($\dot{x}$)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plt.savefig(os.path.join(results_plots, "q1_latent_physics_ode.png"))
plt.close()

# 3. SI Score (The ultimate metric line)
fig, ax = plt.subplots(figsize=(10,5))
time_x = np.arange(len(si_h) + len(si_f))
ax.plot(time_x[:len(si_h)], si_h, label='Healthy Windows', color='green', marker='o', linewidth=2)
ax.plot(time_x[len(si_h):], si_f, label='Fault Impact Evaluation', color='red', marker='s', linewidth=2)
ax.axvline(len(si_h)-0.5, color='gray', linestyle='--')
ax.axhline(0.5, color='black', linestyle=':', label='Change-Point Alarm Boundary', linewidth=2)
ax.set_title("(c) Mahalanobis Z-Score Fusion: Instability Score (SI)")
ax.set_xlabel("Time Progression (CWRU Windows)")
ax.set_ylabel("Instability Score (SI) $\in [0, 1]$")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_plots, "q1_final_SI_score.png"))
plt.close()

# 4. Tabulate Image (Render Table purely with Matplotlib)
res_df = pd.DataFrame({
    'Detection Metric Focus': ['DCN Reconstruction Anomaly', 'Physics ODE Residual Error', 'Final Instability Score (SI)'],
    'Healthy State ($\mu \pm \sigma$)': [f"{np.mean(h_recon):.4f} ± {np.std(h_recon):.4f}", 
                                       f"{np.mean(h_phys):.4f} ± {np.std(h_phys):.4f}", 
                                       f"{np.mean(si_h):.4f} ± {np.std(si_h):.4f}"],
    'Fault Response ($\mu \pm \sigma$)': [f"{np.mean(f_recon):.4f} ± {np.std(f_recon):.4f}", 
                                        f"{np.mean(f_phys):.4f} ± {np.std(f_phys):.4f}", 
                                        f"{np.mean(si_f):.4f} ± {np.std(si_f):.4f}"]
})
res_df.to_csv(os.path.join(results_tables, "phase4_anomaly_metrics.csv"), index=False)

fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=res_df.values, colLabels=res_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4c72b0')
plt.title("Phase 4 Detection Boundary Evaluation Summary", weight='bold', size=14, y=1.1)
plt.savefig(os.path.join(results_tables, "phase4_anomaly_metrics_table.png"), dpi=300, bbox_inches='tight')
plt.close()

print(" SUCCESS: Phase 4 Executed. CWRU Baseline established.")
print(" Results, mathematical boundaries, and visuals output to `results/`.")
