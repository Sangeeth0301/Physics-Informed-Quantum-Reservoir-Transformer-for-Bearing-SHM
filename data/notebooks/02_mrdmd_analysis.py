from scipy.signal import hilbert
from signal_processing import preprocess_bearing_signal
from signal_processing import envelope_signal
def envelope_signal(x):
    analytic = hilbert(x)
    return np.abs(analytic)# ============================================================
# 02_mrdmd_analysis.py
# Phase 2: Standard DMD + mrDMD on CWRU windows
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os
from pydmd import MrDMD, DMD

print(">>> MRDMD PIPELINE STARTING <<<")

# ============================================================
# CONFIG
# ============================================================

processed_dir = r"C:\Users\sange\OneDrive\Desktop\Physics-Informed-Quantum-Reservoir-Transformer\data\processed"
results_dir   = r"C:\Users\sange\OneDrive\Desktop\Physics-Informed-Quantum-Reservoir-Transformer\results\plots"

os.makedirs(results_dir, exist_ok=True)

# ============================================================
# LOAD WINDOWS
# ============================================================

healthy_windows = np.load(os.path.join(processed_dir, "healthy_windows.npy"))
fault_windows   = np.load(os.path.join(processed_dir, "fault_windows.npy"))

print("Healthy windows shape:", healthy_windows.shape)
print("Fault windows shape:", fault_windows.shape)

# DEBUG: visualize envelope effect
from signal_processing import envelope_signal

test_raw = fault_windows[0]
test_env = envelope_signal(test_raw)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.plot(test_raw[:2000], label="Raw", alpha=0.6)
plt.plot(test_env[:2000], label="Envelope", linewidth=2)
plt.legend()
plt.title("Envelope Debug Check")
plt.grid(True)
plt.show()

# ============================================================
# PARAMETERS (research-sensible defaults)
# ============================================================

delay = 60
svd_rank = 12
max_level = 3
max_cycles = 6

# ============================================================
# HANKELIZATION FUNCTION
# ============================================================

def hankelize(signal, delay):
    """Create delay-embedded snapshot matrix for DMD"""
    n = len(signal)
    snapshots = np.zeros((delay, n - delay + 1))
    for i in range(delay):
        snapshots[i, :] = signal[i:i + n - delay + 1]
    return snapshots

# ============================================================
# METRIC EXTRACTION FROM mrDMD
# ============================================================

def extract_dmd_metrics(eigs):
    """Compute research-grade Koopman metrics"""
    if eigs is None or len(eigs) == 0:
        return {
            "spectral_radius": np.nan,
            "unstable_ratio": np.nan,
            "mean_frequency": np.nan
        }

    radii = np.abs(eigs)

    return {
        "spectral_radius": np.max(radii),
        "unstable_ratio": np.sum(radii > 1.0) / len(radii),
        "mean_frequency": np.mean(np.abs(np.imag(eigs)))
    }
# ============================================================
# SAFE mrDMD FITTER (robust research wrapper)
# ============================================================

def safe_mrdmd_fit(signal_hankel, svd_rank, max_level, max_cycles):
    try:
        model = MrDMD(
            DMD(svd_rank=svd_rank),
            max_level=max_level,
            max_cycles=max_cycles
        )
        model.fit(signal_hankel)
        return model.eigs
    except Exception as e:
        print(f"⚠️ mrDMD failure: {e}")
        return np.array([])
# ============================================================
# SELECT ONE WINDOW (debug phase)
# ============================================================

# ============================================================
# SELECT ONE WINDOW (debug phase)
# ============================================================

# ---- pick first window ----
fault_window = fault_windows[0]
healthy_window = healthy_windows[0]

# ---- apply research preprocessing ----
fault_window = preprocess_bearing_signal(fault_window)
healthy_window = preprocess_bearing_signal(healthy_window)

hankel_fault = hankelize(fault_window, delay)
hankel_healthy = hankelize(healthy_window, delay)

print("Hankel fault shape:", hankel_fault.shape)
print("Hankel healthy shape:", hankel_healthy.shape)

# ============================================================
# STANDARD DMD (SANITY CHECK)
# ============================================================

print("\nRunning STANDARD DMD sanity check...")

dmd_test = DMD(svd_rank=15)
dmd_test.fit(hankel_fault)

print("Standard DMD eigenvalues:")
print(dmd_test.eigs)

plt.figure(figsize=(6, 6))
plt.scatter(dmd_test.eigs.real, dmd_test.eigs.imag,
            color='purple', s=50)
plt.title("Standard DMD Eigenvalues (Fault Window)")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(results_dir, "standard_dmd_eigs_fault.png"), dpi=300)
plt.show()

# ============================================================
# mrDMD ON FAULT WINDOW
# ============================================================

print("\nRunning mrDMD on FAULT window...")

mrdmd_fault = MrDMD(
    DMD(svd_rank=svd_rank),
    max_level=max_level,
    max_cycles=max_cycles
)
mrdmd_fault.fit(hankel_fault)

print("Fault mrDMD eigenvalues:", mrdmd_fault.eigs)

# ============================================================
# SAFE MODE PLOTTING (FAULT)
# ============================================================

modes_fault = mrdmd_fault.modes

if modes_fault.shape[1] == 0:
    print("⚠️ No mrDMD modes found for this fault window — skipping mode plot.")
else:
    fig_modes, axs_modes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axs_modes[0].plot(np.real(modes_fault[:, 0]),
                      color='blue', label='Fault Mode 0')
    axs_modes[0].set_title("mrDMD Mode 0 (Fault)")
    axs_modes[0].legend()
    axs_modes[0].grid(True, alpha=0.3)

    if modes_fault.shape[1] > 1:
        axs_modes[1].plot(np.real(modes_fault[:, 1]),
                          color='blue', label='Fault Mode 1')
        axs_modes[1].set_title("mrDMD Mode 1 (Fault)")
    else:
        axs_modes[1].text(0.5, 0.5, "Only one mode found",
                          transform=axs_modes[1].transAxes,
                          ha='center')

    axs_modes[1].set_xlabel("Sample")
    axs_modes[1].legend()
    axs_modes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mrdmd_modes_fault.png"), dpi=300)
    plt.show()

# ============================================================
# mrDMD ON HEALTHY WINDOW
# ============================================================

print("\nRunning mrDMD on HEALTHY window...")

mrdmd_healthy = MrDMD(
    DMD(svd_rank=svd_rank),
    max_level=max_level,
    max_cycles=max_cycles
)
mrdmd_healthy.fit(hankel_healthy)

print("Healthy mrDMD eigenvalues:", mrdmd_healthy.eigs)

# ============================================================
# MODE COMPARISON (SAFE)
# ============================================================

if (mrdmd_fault.modes.shape[1] > 0) and (mrdmd_healthy.modes.shape[1] > 0):
    plt.figure(figsize=(12, 5))
    plt.plot(np.real(mrdmd_healthy.modes[:, 0]),
             color='green', label='Healthy Mode 0', alpha=0.8)
    plt.plot(np.real(mrdmd_fault.modes[:, 0]),
             color='red', label='Fault Mode 0', alpha=0.8)

    plt.title("mrDMD Mode Comparison: Healthy vs Fault")
    plt.xlabel("Sample")
    plt.ylabel("Mode Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(results_dir, "mrdmd_mode_comparison.png"), dpi=300)
    plt.show()
else:
    print("⚠️ Mode comparison skipped due to missing modes.")

print("\n✅ SINGLE-WINDOW MRDMD COMPLETED.")

# ============================================================
# MULTI-WINDOW MRDMD ANALYSIS (RESEARCH MODE)
# ============================================================

print("\n>>> Running multi-window mrDMD analysis <<<")

num_windows = min(80, len(fault_windows), len(healthy_windows))

fault_metrics = []
healthy_metrics = []

for i in range(num_windows):

    # ---------------- FAULT ----------------
    fw = preprocess_bearing_signal(fault_windows[i])
    hankel_f = hankelize(fw, delay)

    eigs_fault = safe_mrdmd_fit(
        hankel_f, svd_rank, max_level, max_cycles
    )
    fault_metrics.append(extract_dmd_metrics(eigs_fault))


# ---------------- HEALTHY ----------------
    hw = preprocess_bearing_signal(healthy_windows[i])
    hankel_h = hankelize(hw, delay)

    eigs_healthy = safe_mrdmd_fit(
        hankel_h, svd_rank, max_level, max_cycles
    )
    healthy_metrics.append(extract_dmd_metrics(eigs_healthy))
# ============================================================
# STATISTICAL SUMMARY
# ============================================================

def summarize(metric_list, key):
    vals = [m[key] for m in metric_list if not np.isnan(m[key])]
    return np.mean(vals), np.std(vals)

print("\n=== Koopman Statistical Summary ===")

for key in ["spectral_radius", "unstable_ratio", "mean_frequency"]:
    h_mean, h_std = summarize(healthy_metrics, key)
    f_mean, f_std = summarize(fault_metrics, key)

    print(f"\nMetric: {key}")
    print(f"Healthy → mean: {h_mean:.4f}, std: {h_std:.4f}")
    print(f"Fault   → mean: {f_mean:.4f}, std: {f_std:.4f}")

print("\n🚀 MRDMD RESEARCH PIPELINE FINISHED.")