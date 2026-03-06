import numpy as np
import torch
import torch.nn as nn
from src.data_prep.signal_processing import preprocess_bearing_signal
from pydmd import DMD, MrDMD
from src.quantum.pqkr import PQKR

class FrozenDCNEncoder(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ELU(),
            nn.Linear(32, 16), nn.ELU(),
            nn.Linear(16, 8)
        )
    def forward(self, x):
        return self.encoder(x)

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

def extract_latent_trajectory(windows, scaler, pca, dcn_encoder, pqkr=None):
    """
    Extracts deterministic latent trajectories from raw windows.
    Requires pre-fitted standard scaler, PCA, and a frozen DCN encoder.
    """
    if pqkr is None:
        pqkr = PQKR(n_qubits=5, n_layers=2, seed=42)
        
    f_base = np.array([extract_mrdmd_features(w) for w in windows])
    x_norm = scaler.transform(f_base)
    x_pca = pca.transform(x_norm)
    
    q_states = np.array([np.concatenate([np.real(pqkr.get_state(x)), np.imag(pqkr.get_state(x))]) for x in x_pca])
    
    with torch.no_grad():
        t_states = torch.tensor(q_states, dtype=torch.float32)
        z_latent = dcn_encoder(t_states).numpy()
    
    return z_latent
