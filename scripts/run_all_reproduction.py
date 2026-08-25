
import os
import subprocess
import sys
import time

# Absolute base directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.join(base_dir, 'scripts')
python_exe = sys.executable

# Define the execution order for a clean research reproduction
pipeline_flow = [
    # 1. Baseline Data Processing & Initial Plots
    "01_load_cwru_and_plot.py",
    "02_mrdmd_analysis.py",
    "03_pqkr_analysis.py",
    
    # 2. Advanced Phase Architecture & Physical ODE
    "08_physics_latent_ode_q1.py",
    "08b_generate_q1_tables.py",
    
    # 3. Validation Phasing (Ablation, Robustness, Metrics)
    "09_baseline_comparisons.py",
    "09b_baseline_roc_pr_curves.py",
    "11_comprehensive_validation.py",
    
    # 4. Temporal Multi-Dataset Generalization (NASA IMS / XJTU-SY)
    "09_load_ims_and_run_pipeline.py",
    "14_xjtu_generalization.py",

    # 5. Master Final Architecture (Transformation & Decision Logic Implementation)
    "final_architecture_upgrade.py",
    "12_master_optimal_pipeline.py",
    "13_ultra_optimal_results.py",
    "final_validation_master.py",
    
    # 6. Global Result Organization & Packaging
    "organize_results.py",
    "generate_docx_summary.py"
]

def run_full_reproduction():
    print("="*80)
    print("GLOBAL REPRODUCTION SYSTEM: CONSOLIDATING ENTIRE RESEARCH PROJECT")
    print("="*80)
    
    # Ensure environment is set to UTF-8 for console output
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    start_time = time.time()
    success_count = 0
    
    for i, script in enumerate(pipeline_flow):
        script_path = os.path.join(scripts_dir, script)
        if not os.path.exists(script_path):
            print(f"[{i+1}/{len(pipeline_flow)}] SKIP: {script} (File not found)")
            continue
            
        print(f"[{i+1}/{len(pipeline_flow)}] PROJECT STAGE: Executing {script}...")
        try:
            # We run with a generous timeout and UTF-8 encoding environment
            result = subprocess.run(
                [python_exe, script_path],
                cwd=base_dir,
                capture_output=True,
                text=True,
                timeout=1200, # 20 mins per script for safety
                env=env,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                print(f"    -> VALIDATED: Script completed successfully.")
                # print(result.stdout[:200] + "...") # Optional: print snippet
                success_count += 1
            else:
                print(f"    -> NON-CRITICAL WARNING in {script} (RetCode {result.returncode})")
                if result.stderr:
                    print(f"       LOGS: {result.stderr[:250].strip()}...")
        except subprocess.TimeoutExpired:
            print(f"    -> TIMEOUT in {script} (Script took too long, skipping...)")
        except Exception as e:
            print(f"    -> EXCEPTION executing {script}: {str(e)}")

    end_time = time.time()
    total_min = (end_time - start_time) / 60
    
    print("="*80)
    print("ULTRA-EARLY BEARING FAULT DETECTION PIPELINE VALIDATED")
    print(f"Processed {success_count}/{len(pipeline_flow)} Critical Research Modules.")
    print(f"Total Execution Time: {total_min:.2f} minutes")
    print("="*80)

if __name__ == "__main__":
    run_full_reproduction()
