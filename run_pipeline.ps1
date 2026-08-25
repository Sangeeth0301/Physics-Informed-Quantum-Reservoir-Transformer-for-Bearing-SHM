
# Master Run Script Wrap for Q1 Project
# This triggers the consolidation runner for the Physics-Informed Quantum Reservoir Transformer.

$PythonPath = ".\.venv\Scripts\python.exe"
if (-Not (Test-Path $PythonPath)) {
    $PythonPath = "python"
}

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "STARTING GLOBAL RESEARCH REPRODUCTION" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

& $PythonPath "scripts\run_all_reproduction.py"

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "RESEARCH PIPELINE COMPLETE: RESULTS SAVED IN /results/" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan
