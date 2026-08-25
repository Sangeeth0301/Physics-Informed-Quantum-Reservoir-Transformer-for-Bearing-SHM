# Scripts Navigation Guide

This directory contains the numbered research pipeline scripts. Run them in the order below to reproduce all results.

---

## 🚀 Quick Reproduction

```bash
# Run the complete end-to-end pipeline
python scripts/run_all_reproduction.py
```

---

## 📋 Script Reference Table

| Script | Stage | Purpose | Input | Output |
|:---:|:---:|---|---|---|
| `01_load_cwru_and_plot.py` | **Phase 1** | Load CWRU .mat files, plot healthy vs fault signals | `data/raw/*.mat` | Plots + `data/processed/*.npy` |
| `02_mrdmd_analysis.py` | **Phase 2** | Multi-resolution DMD, Koopman eigenvalues, spectral radius | `data/processed/*.npy` | mrDMD features, eigenvalue plots |
| `03_pqkr_analysis.py` | **Phase 3** | Quantum kernel reservoir encoding + fidelity kernel | mrDMD features | Quantum divergence scores, UMAP |
| `04_dcn_physics_ode.py` | **Phase 4** | Dynamical Consistency Network + Physics ODE integration | Quantum features | Physics residuals, latent ODE |
| `04.5_validation_ablation.py` | **Phase 4** | Ablation study — component contributions | All features | Ablation table |
| `04.6_fusion_diagnostics.py` | **Phase 4** | Diagnostics on fusion layer | Fused features | Diagnostic plots |
| `04.7_final_statistical_hardening.py` | **Phase 4** | Statistical hardening (10-seed) | Pipeline outputs | Hardened metrics |
| `04_q1_publication_graphics.py` | **Phase 4** | Publication-quality figures | Results | Q1-grade plots |
| `05_physics_latent_ode.py` | **Phase 5** | Physics-constrained latent ODE trajectories | Latent space | ODE trajectories |
| `05_q1_statistical_hardening.py` | **Phase 5** | Q1-grade statistical hardening | Pipeline outputs | Hardened tables |
| `06_export_optimal_results.py` | **Export** | Export best results to `results/` | All outputs | Final artifacts |
| `07_phase3_hardening.py` | **Phase 3** | Phase 3 statistical robustness | PQKR outputs | Robustness report |
| `08_phase3_final_robustness.py` | **Phase 3** | Final robustness validation | Hardened outputs | Validation plots |
| `08_physics_latent_ode_q1.py` | **Phase 5** | Q1-grade physics ODE version | Latent space | Q1 ODE metrics |
| `08_physics_latent_ode_v2.py` | **Phase 5** | ODE v2 with improved solver | Latent space | Improved trajectories |
| `08b_generate_q1_tables.py` | **Export** | Generate Q1 LaTeX/CSV tables | Results | LaTeX tables |
| `08c_generate_q1_pinn_table.py` | **Export** | PINN-specific Q1 table | PINN results | PINN LaTeX table |
| `09_baseline_comparisons.py` | **Validation** | Classical baseline comparison | All features | Baseline table |
| `09_load_ims_and_run_pipeline.py` | **IMS** | Load IMS NASA dataset and run pipeline | `data/raw/IMS/` | IMS results |
| `09b_baseline_roc_pr_curves.py` | **Validation** | ROC and PR curves for baselines | Baseline scores | ROC/PR plots |
| `11_comprehensive_validation.py` | **Validation** | Full comprehensive validation | All outputs | Validation report |
| `12_master_optimal_pipeline.py` | **Master** | **Master pipeline — best configuration** | Raw data | Final SI score |
| `13_ultra_optimal_results.py` | **Master** | Ultra-optimized result generation | Pipeline | Best results |
| `14_xjtu_generalization.py` | **XJTU** | XJTU-SY dataset generalization | `data/raw/XJTU/` | Generalization results |
| `final_architecture_upgrade.py` | **Upgrade** | Architecture upgrade script | Previous outputs | Upgraded model |
| `final_validation_master.py` | **Validation** | Master validation of final model | Final model | Validation metrics |
| `run_all_reproduction.py` | **Meta** | **Full end-to-end reproduction** | `data/raw/` | All results |
| `organize_results.py` | **Utility** | Organize results into structured folders | `results/` | Organized `results/` |
| `99_finalize_results_and_docx.py` | **Export** | Finalize results and generate DOCX report | All results | Report DOCX |

---

## 🗂️ Recommended Run Order for Reproduction

```
Phase 1  →  01_load_cwru_and_plot.py
Phase 2  →  02_mrdmd_analysis.py
Phase 3  →  03_pqkr_analysis.py  →  07_phase3_hardening.py
Phase 4  →  04_dcn_physics_ode.py  →  04.5_validation_ablation.py
Phase 5  →  05_physics_latent_ode.py
Master   →  12_master_optimal_pipeline.py
Export   →  06_export_optimal_results.py  →  08b_generate_q1_tables.py
```

---

## 📊 Output Locations

| Output Type | Location |
|---|---|
| Raw numpy tensors | `results/01_data_arrays/` |
| Statistical tables (CSV + LaTeX) | `results/02_statistical_tables/` |
| Publication figures | `results/03_publication_figures/` |
| Processed data | `data/processed/` |
