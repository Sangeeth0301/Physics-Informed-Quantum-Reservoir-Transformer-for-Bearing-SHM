# Complete robust CWRU loader + plot (handles v7.3 HDF5 files)

import scipy.io as sio
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIG ===
data_dir = r"C:\Users\sange\OneDrive\Desktop\Physics-Informed-Quantum-Reservoir-Transformer\data\raw\CWRU"

healthy_file = "97.mat"   # Healthy baseline (0 HP)
fault_file   = "107.mat"  # 7 mil outer race fault

# === Super robust loader ===
def load_de_time(file_path):
    print(f"\nLoading file: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
    
    try:
        # Attempt 1: scipy (classic .mat)
        mat = sio.loadmat(file_path)
        print("scipy keys:", list(mat.keys()))
        
        de_keys = [k for k in mat.keys() if 'DE_time' in k or 'BA_time' in k or 'FE_time' in k]
        if de_keys:
            key = de_keys[0]
            signal = mat[key].flatten().astype(np.float32)
            print(f"Success (scipy): key = {key}, shape = {signal.shape}")
            return signal

        # Attempt 2: h5py (v7.3 HDF5)
        print("scipy failed → trying h5py...")
        with h5py.File(file_path, 'r') as f:
            print("h5py top-level keys:", list(f.keys()))
            for group in f:
                if isinstance(f[group], h5py.Group):
                    print(f"  Group '{group}' keys:", list(f[group].keys()))
                    for key in f[group]:
                        if 'DE_time' in key or 'BA_time' in key or 'FE_time' in key:
                            signal = np.array(f[group][key]).flatten().astype(np.float32)
                            print(f"Success (h5py): {group}/{key}, shape = {signal.shape}")
                            return signal

        raise ValueError("No DE/BA/FE_time key found")

    except Exception as e:
        print(f"ERROR loading {file_path}: {str(e)}")
        raise

# === Load ===
healthy_path = os.path.join(data_dir, healthy_file)
fault_path   = os.path.join(data_dir, fault_file)

healthy_signal = load_de_time(healthy_path)
fault_signal   = load_de_time(fault_path)

print(f"\nHealthy shape: {healthy_signal.shape}")
print(f"Fault shape:   {fault_signal.shape}")
print("Sampling rate: 12000 Hz (CWRU 12 kHz)")

# === Full plot ===
fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True, dpi=100)

axs[0].plot(healthy_signal, color='green', lw=1, alpha=0.9, label=f'Healthy ({healthy_file})')
axs[0].set_title('Healthy Baseline – Drive End Acceleration', fontsize=14)
axs[0].set_ylabel('Acceleration (g)', fontsize=12)
axs[0].legend()
axs[0].grid(True, alpha=0.3)

axs[1].plot(fault_signal, color='red', lw=1, alpha=0.9, label=f'7 mil Fault ({fault_file})')
axs[1].set_title('Incipient Fault Signal – Drive End Acceleration', fontsize=14)
axs[1].set_xlabel('Sample Index', fontsize=12)
axs[1].set_ylabel('Acceleration (g)', fontsize=12)
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Zoom plot (first 10k samples – impulses should be visible here) ===
zoom_start, zoom_end = 0, 10000

fig_zoom, axs_zoom = plt.subplots(2, 1, figsize=(14, 8), sharex=True, dpi=100)

axs_zoom[0].plot(healthy_signal[zoom_start:zoom_end], color='green', lw=1.5)
axs_zoom[0].set_title(f'Zoom: Healthy (samples 0–{zoom_end})', fontsize=14)
axs_zoom[0].set_ylabel('Acceleration (g)')
axs_zoom[0].grid(True, alpha=0.3)

axs_zoom[1].plot(fault_signal[zoom_start:zoom_end], color='red', lw=1.5)
axs_zoom[1].set_title(f'Zoom: 7 mil Fault (samples 0–{zoom_end})', fontsize=14)
axs_zoom[1].set_xlabel('Sample Index')
axs_zoom[1].set_ylabel('Acceleration (g)')
axs_zoom[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("If you see periodic spikes in the red zoom plot → success! Early fault impulses are visible.")
# --- Basic Preprocessing & Windowing ---

window_size = 2048  # common choice for bearing analysis
stride = 512        # overlapping windows

def segment_signal(signal, window_size, stride):
    segments = []
    for start in range(0, len(signal) - window_size + 1, stride):
        segments.append(signal[start:start + window_size])
    return np.array(segments)

# Segment healthy
healthy_segments = segment_signal(healthy_signal, window_size, stride)
print(f"Healthy segments shape: {healthy_segments.shape}")

# Segment fault
fault_segments = segment_signal(fault_signal, window_size, stride)
print(f"Fault segments shape: {fault_segments.shape}")

# Optional: Plot one window from each
fig_win, axs_win = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axs_win[0].plot(healthy_segments[0], color='green', lw=1.2)
axs_win[0].set_title('One Healthy Window (2048 samples)')
axs_win[0].grid(True, alpha=0.3)

axs_win[1].plot(fault_segments[0], color='red', lw=1.2)
axs_win[1].set_title('One Fault Window (2048 samples)')
axs_win[1].set_xlabel('Sample within window')
axs_win[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save processed data (fast loading later)
processed_dir = r"C:\Users\sange\OneDrive\Desktop\Physics-Informed-Quantum-Reservoir-Transformer\data\processed"
os.makedirs(processed_dir, exist_ok=True)

np.save(os.path.join(processed_dir, "healthy_windows.npy"), healthy_segments)
np.save(os.path.join(processed_dir, "fault_windows.npy"), fault_segments)

print("Processed windows saved to data/processed/")
# --- Preprocessing & Windowing (prepare for mrDMD & models) ---

window_size = 2048  # standard for bearing analysis (2048 samples ≈ 0.17 s at 12 kHz)
stride = 512        # 75% overlap – good for capturing transients

def segment_signal(signal, window_size, stride):
    segments = []
    for start in range(0, len(signal) - window_size + 1, stride):
        segments.append(signal[start:start + window_size])
    return np.array(segments)

# Segment both signals
healthy_windows = segment_signal(healthy_signal, window_size, stride)
fault_windows   = segment_signal(fault_signal, window_size, stride)

print(f"Healthy windows shape: {healthy_windows.shape}  (num_windows, window_size)")
print(f"Fault windows shape:   {fault_windows.shape}")

# Optional: Plot one random window from each
fig_win, axs_win = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Pick first window
axs_win[0].plot(healthy_windows[0], color='green', lw=1.2)
axs_win[0].set_title('Example Healthy Window (2048 samples)')
axs_win[0].set_ylabel('Acceleration (g)')
axs_win[0].grid(True, alpha=0.3)

axs_win[1].plot(fault_windows[0], color='red', lw=1.2)
axs_win[1].set_title('Example Fault Window (2048 samples)')
axs_win[1].set_xlabel('Sample within window')
axs_win[1].set_ylabel('Acceleration (g)')
axs_win[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save processed windows as .npy (fast loading later)
processed_dir = r"C:\Users\sange\OneDrive\Desktop\Physics-Informed-Quantum-Reservoir-Transformer\data\processed"
os.makedirs(processed_dir, exist_ok=True)

np.save(os.path.join(processed_dir, "healthy_windows.npy"), healthy_windows)
np.save(os.path.join(processed_dir, "fault_windows.npy"), fault_windows)

print("Processed windows saved to data/processed/")
# --- Preprocessing & Window Saving (prepare for mrDMD) ---

window_size = 2048
stride = 512  # 75% overlap

def segment_signal(signal, window_size, stride):
    segments = []
    for start in range(0, len(signal) - window_size + 1, stride):
        segments.append(signal[start:start + window_size])
    return np.array(segments)

healthy_windows = segment_signal(healthy_signal, window_size, stride)
fault_windows   = segment_signal(fault_signal, window_size, stride)

print(f"Healthy windows: {healthy_windows.shape}")
print(f"Fault windows:   {fault_windows.shape}")

# Save as .npy (fast loading later)
processed_dir = r"C:\Users\sange\OneDrive\Desktop\Physics-Informed-Quantum-Reservoir-Transformer\data\processed"
os.makedirs(processed_dir, exist_ok=True)

np.save(os.path.join(processed_dir, "healthy_windows.npy"), healthy_windows)
np.save(os.path.join(processed_dir, "fault_windows.npy"), fault_windows)

print("Windows saved to data/processed/")