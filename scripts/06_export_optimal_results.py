import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt

# Define paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
plots_dir = os.path.join(base_dir, 'results', 'plots')
tables_dir = os.path.join(base_dir, 'results', 'tables')
optimal_dir = os.path.join(base_dir, 'results', 'optimal_q1_gallery')

os.makedirs(optimal_dir, exist_ok=True)

print("=======================================================================")
print("EXPORTING OPTIMAL Q1 RESULTS AND RENDERING TABLES AS IMAGES            ")
print("=======================================================================\n")

# 1. Copy Optimal Plots
optimal_plots = [
    "q1_2D_fidelity_heatmap_annotated.png",
    "q1_log_eigenvalue_decay_benched.png",
    "q1_umap_quantum_projection.png",
    "phase_space_attractor_3d.png",
    "koopman_unit_circle_kde_fault.png"
]

print("[*] Copying optimal publication-ready graphics...")
for plot in optimal_plots:
    src = os.path.join(plots_dir, plot)
    dst = os.path.join(optimal_dir, plot)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  -> Copied {plot}")
    else:
        print(f"  -> [WARNING] {plot} not found!")

# 2. Render Tables as Images
def render_mpl_table(data, filename, col_width=3.0, row_height=0.625, font_size=10,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

import numpy as np

print("\n[*] Rendering Tabular Data as High-Res Images...")

# Table 1: Koopman Metrics
if os.path.exists(os.path.join(tables_dir, "koopman_metrics.csv")):
    df_koopman = pd.read_csv(os.path.join(tables_dir, "koopman_metrics.csv"))
    render_mpl_table(df_koopman, os.path.join(optimal_dir, "table_1_koopman_metrics.png"), col_width=4.0)
    print("  -> Rendered Koopman Metrics table")

# Table 2: Ablation Grid
if os.path.exists(os.path.join(tables_dir, "hyperparameter_ablation_grid.csv")):
    df_ablation = pd.read_csv(os.path.join(tables_dir, "hyperparameter_ablation_grid.csv"))
    # Format floats for cleaner rendering
    render_mpl_table(df_ablation, os.path.join(optimal_dir, "table_2_ablation_grid.png"), col_width=2.5, font_size=9)
    print("  -> Rendered Ablation Grid table")

# Table 3: Divergence Metrics
if os.path.exists(os.path.join(tables_dir, "divergence_metrics.csv")):
    df_divergence = pd.read_csv(os.path.join(tables_dir, "divergence_metrics.csv"))
    render_mpl_table(df_divergence, os.path.join(optimal_dir, "table_3_divergence_metrics.png"), col_width=4.5)
    print("  -> Rendered Divergence Metrics table")

print("\n🚀 OPTIMAL RESULTS VAULT CREATED SUCCESSFULLY.")
