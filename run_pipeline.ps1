$scripts = @(
    "scripts\01_load_cwru_and_plot.py",
    "scripts\02_mrdmd_analysis.py",
    "scripts\03_pqkr_analysis.py",
    "scripts\08_phase3_final_robustness.py",
    "scripts\04_q1_publication_graphics.py",
    "scripts\04.5_validation_ablation.py",
    "scripts\04.6_fusion_diagnostics.py",
    "scripts\04.7_final_statistical_hardening.py",
    "scripts\05_physics_latent_ode.py",
    "scripts\09_load_ims_and_run_pipeline.py",
    "scripts\06_export_optimal_results.py"
)

foreach ($script in $scripts) {
    Write-Host "`n======================================================="
    Write-Host "Running $script"
    Write-Host "======================================================="
    & .\.venv\Scripts\python.exe $script
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Script $script failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}
Write-Host "`n======================================================="
Write-Host "PIPELINE COMPLETED SUCCESSFULLY"
Write-Host "======================================================="
