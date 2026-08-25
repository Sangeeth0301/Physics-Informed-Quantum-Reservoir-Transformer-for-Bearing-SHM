import os
import shutil

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

# Target directories
dirs = {
    "arrays": os.path.join(base_dir, "01_data_arrays"),
    "csv_tex": os.path.join(base_dir, "02_statistical_tables"),
    "q1_figs_classical": os.path.join(base_dir, "03_publication_figures", "A_Classical_MrDMD"),
    "q1_figs_quantum": os.path.join(base_dir, "03_publication_figures", "B_Quantum_PQKR"),
    "q1_figs_dl": os.path.join(base_dir, "03_publication_figures", "C_DeepLearning_DCN"),
    "q1_figs_physics": os.path.join(base_dir, "03_publication_figures", "D_Physics_PINN"),
    "q1_figs_tables": os.path.join(base_dir, "03_publication_figures", "E_Formal_Tables"),
    "q1_figs_general": os.path.join(base_dir, "03_publication_figures", "F_General_Robustness")
}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)

# Iterate through ALL files in results and organize them
for root, _, files in os.walk(base_dir):
    # Skip the new organized folders to prevent recursive loops
    if any(new_dir in root for new_dir in dirs.values()):
        continue
        
    for file in files:
        src = os.path.join(root, file)
        ext = file.lower().split('.')[-1]
        name = file.lower()
        
        # Determine destination
        dst_dir = None
        
        if ext == "npy":
            dst_dir = dirs["arrays"]
        elif ext in ["csv", "tex"]:
            dst_dir = dirs["csv_tex"]
        elif ext == "md":
            continue # Leave markdown summaries alone or handle them separately. Actually let's keep them in root.
        elif ext in ["png", "jpg", "jpeg", "svg"]:
            # Image categorization
            if "table" in name:
                dst_dir = dirs["q1_figs_tables"]
            elif any(k in name for k in ["pinn", "physics", "latent_trajectory", "residual", "ode"]):
                dst_dir = dirs["q1_figs_physics"]
            elif any(k in name for k in ["quantum", "pqkr", "qubit", "kernel", "fidelity", "hilbert"]):
                dst_dir = dirs["q1_figs_quantum"]
            elif any(k in name for k in ["mrdmd", "koopman", "eigen", "phase_space", "standard_dmd"]):
                dst_dir = dirs["q1_figs_classical"]
            elif any(k in name for k in ["dcn", "si_score", "roc", "pr", "baseline"]):
                dst_dir = dirs["q1_figs_dl"]
            else:
                dst_dir = dirs["q1_figs_general"]
        
        if dst_dir:
            dst = os.path.join(dst_dir, file)
            # if the file already exists in the destination, overwrite it to have the latest
            if os.path.exists(dst):
                os.remove(dst)
            shutil.copy2(src, dst)

print("Organization Complete.")
