---
description: Run the complete Physics-Informed Quantum Reservoir Transformer pipeline
---
This workflow automates the chronological execution of the entire Q1-grade pipeline, from raw signal extraction up to the Physics Neural ODE scoring. 

// turbo-all

1. Phase 1: Signal preparation and spectrogram visualization
```bash
.\.venv\Scripts\python.exe scripts\01_load_cwru_and_plot.py
```

2. Phase 2: Multi-Resolution Dynamic Mode Decomposition (Classical Features)
```bash
.\.venv\Scripts\python.exe scripts\02_mrdmd_analysis.py
```

3. Phase 3: Projected Quantum Kernel Reservoir (PQKR) Evaluation
```bash
.\.venv\Scripts\python.exe scripts\03_pqkr_analysis.py
```

4. Phase 3.5: Quantum Kernel Robustness and Cross-Validation
```bash
.\.venv\Scripts\python.exe scripts\08_phase3_final_robustness.py
```

5. Phase 4: Dynamical Consistency Network (DCN) & Baseline SI Generation
```bash
.\.venv\Scripts\python.exe scripts\04_q1_publication_graphics.py
```

6. Phase 4.5: System-Level Validation, Noise Robustness, and Ablation
```bash
.\.venv\Scripts\python.exe scripts\04.5_validation_ablation.py
```

7. Phase 4.6: Diagnostic Fusion Fix (Resolving DCN Dominance)
```bash
.\.venv\Scripts\python.exe scripts\04.6_fusion_diagnostics.py
```

8. Phase 4.7: Final Multi-Seed Statistical Hardening (PR-AUC)
```bash
.\.venv\Scripts\python.exe scripts\04.7_final_statistical_hardening.py
```

9. Phase 5: Physics-Guided Latent ODE (Soft Jeffcott Constraint)
```bash
.\.venv\Scripts\python.exe scripts\05_physics_latent_ode.py
```

10. Phase 5.5: NASA IMS Temporal Translation (35-Day Progression)
```bash
.\.venv\Scripts\python.exe scripts\09_load_ims_and_run_pipeline.py
```

11. Clean Up / Final Export formatting
```bash
.\.venv\Scripts\python.exe scripts\06_export_optimal_results.py
```
