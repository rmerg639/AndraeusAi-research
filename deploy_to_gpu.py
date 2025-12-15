#!/usr/bin/env python3
"""
Deploy experiment code to cloud GPU.

Usage:
    python deploy_to_gpu.py --upload
    python deploy_to_gpu.py --run-comparison
    python deploy_to_gpu.py --run-experiments
    python deploy_to_gpu.py --status

Copyright (c) 2025 Rocco Andraeus Sergi. All Rights Reserved.
"""

import subprocess
import sys
import os
from pathlib import Path

# =============================================================================
# GPU SERVER CONFIG - Load from environment variables
# =============================================================================

GPU_HOST = os.environ.get("ANDRAEUS_GPU_HOST", "")
GPU_PORT = os.environ.get("ANDRAEUS_GPU_PORT", "22")
GPU_USER = os.environ.get("ANDRAEUS_GPU_USER", "")
REMOTE_DIR = os.environ.get("ANDRAEUS_REMOTE_DIR", "/root/andraeus-research")

# SSH command base (built dynamically)
SSH_CMD = f"ssh -p {GPU_PORT} {GPU_USER}@{GPU_HOST}" if GPU_HOST else ""
SCP_CMD = f"scp -P {GPU_PORT}" if GPU_HOST else ""


def _validate_gpu_config():
    """Validate GPU configuration is set before operations."""
    if not GPU_HOST or not GPU_USER:
        print("\n" + "="*60)
        print("  ERROR: GPU credentials not configured!")
        print("="*60)
        print("\nSet environment variables:")
        print("  export ANDRAEUS_GPU_HOST=your.server.ip")
        print("  export ANDRAEUS_GPU_PORT=22")
        print("  export ANDRAEUS_GPU_USER=your_username")
        print("\nOr create a .env file from .env.example:")
        print("  cp .env.example .env")
        print("  # Edit .env with your credentials")
        print("="*60 + "\n")
        sys.exit(1)

# =============================================================================
# DEPLOYMENT FUNCTIONS
# =============================================================================

def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
    return result

def check_connection():
    """Check if we can connect to the GPU server."""
    _validate_gpu_config()  # Ensure credentials are set
    print("Checking connection to GPU server...")
    result = run_cmd(f'{SSH_CMD} "echo Connected successfully"', check=False)
    return result.returncode == 0

def upload_files():
    """Upload experiment files to GPU server."""
    print("\n" + "="*50)
    print("  UPLOADING FILES TO GPU SERVER")
    print("="*50)

    # Create remote directory
    run_cmd(f'{SSH_CMD} "mkdir -p {REMOTE_DIR}/evaluation"')
    run_cmd(f'{SSH_CMD} "mkdir -p {REMOTE_DIR}/extensions"')
    run_cmd(f'{SSH_CMD} "mkdir -p {REMOTE_DIR}/output"')

    # Files to upload
    local_dir = Path(__file__).parent

    files_to_upload = [
        # Core training
        "train_personal_ai.py",
        "user_config.py",
        # Evaluation
        "evaluation/eval_framework.py",
        "evaluation/ablation_study.py",
        "evaluation/baseline_rag.py",
        "evaluation/depth_experiment.py",
        "evaluation/run_experiments.py",
        "evaluation/before_after_comparison.py",
        "evaluation/results_tracker.py",
        "evaluation/results_template.csv",
        "evaluation/results_summary.csv",
        "evaluation/METHODOLOGY.md",
        # Extensions
        "extensions/professional_config.py",
        "extensions/live_context_server.py",
    ]

    for f in files_to_upload:
        local_path = local_dir / f
        if local_path.exists():
            remote_path = f"{GPU_USER}@{GPU_HOST}:{REMOTE_DIR}/{f}"
            run_cmd(f'{SCP_CMD} "{local_path}" {remote_path}')
            print(f"  ✓ Uploaded {f}")
        else:
            print(f"  ✗ Missing {f}")

    print("\nUpload complete!")

def install_dependencies():
    """Install required packages on GPU server."""
    print("\n" + "="*50)
    print("  INSTALLING DEPENDENCIES")
    print("="*50)

    deps = [
        "pip install torch transformers accelerate peft bitsandbytes",
        "pip install datasets trl",
        "pip install pandas openpyxl",  # For Excel export
    ]

    for cmd in deps:
        run_cmd(f'{SSH_CMD} "{cmd}"')

def run_before_after_comparison():
    """Run the before/after comparison on GPU server."""
    print("\n" + "="*50)
    print("  RUNNING BEFORE/AFTER COMPARISON")
    print("="*50)

    # Check if adapter exists
    print("Checking for existing adapter...")
    result = run_cmd(f'{SSH_CMD} "ls -la {REMOTE_DIR}/output/"', check=False)

    # Run comparison
    script = f"""
cd {REMOTE_DIR}
python -c "
from evaluation.before_after_comparison import *

# Configure user
user_config = {{
    'user_name': 'User',
    'user_age': '25',
    'user_birthday': 'January 1',
    'pet_name': 'Buddy',
    'pet_type': 'dog',
    'pet_breed': 'Golden Retriever',
    'ai_name': 'Assistant',
}}

# Check for adapter
import os
adapter_path = './output/personal-ai' if os.path.exists('./output/personal-ai') else None

if adapter_path is None:
    print('No adapter found! Train first with train_personal_ai.py')
    print('Running base model only comparison...')

# Run comparison
comparison = BeforeAfterComparison(
    base_model_name='Qwen/Qwen2.5-7B-Instruct',
    adapter_path=adapter_path
)

print('Loading models (this takes 1-2 minutes)...')
comparison.load_models()

questions = get_comparison_questions(user_config)
system_prompt = 'You are a helpful AI assistant.'

summary = comparison.run_comparison(questions, system_prompt)
print_comparison_report(summary)
save_comparison(summary, 'evaluation/comparison_results.json')
print(generate_comparison_table(summary))
"
"""
    run_cmd(f'{SSH_CMD} \'{script}\'')

def run_full_experiments():
    """Run the full experiment suite on GPU server."""
    print("\n" + "="*50)
    print("  RUNNING FULL EXPERIMENT SUITE")
    print("="*50)
    print("WARNING: This will cost [amount] and take ~9 hours!")
    print("Make sure you have the budget and time.\n")

    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    # Run experiments (in background with nohup)
    script = f"""
cd {REMOTE_DIR}
# Edit run_experiments.py to set dry_run=False
sed -i 's/dry_run=True/dry_run=False/g' evaluation/run_experiments.py

# Run with nohup so it continues if SSH disconnects
nohup python evaluation/run_experiments.py > experiment_output.log 2>&1 &
echo "Experiments started in background. Check experiment_output.log for progress."
echo "PID: $!"
"""
    run_cmd(f'{SSH_CMD} \'{script}\'')

def check_status():
    """Check experiment status on GPU server."""
    print("\n" + "="*50)
    print("  CHECKING EXPERIMENT STATUS")
    print("="*50)

    # Check if experiments are running
    run_cmd(f'{SSH_CMD} "pgrep -f run_experiments.py && echo \'Experiments running\' || echo \'No experiments running\'"')

    # Show recent log output
    print("\nRecent log output:")
    run_cmd(f'{SSH_CMD} "tail -50 {REMOTE_DIR}/experiment_output.log 2>/dev/null || echo \'No log file yet\'"')

    # Show results summary
    print("\nResults summary:")
    run_cmd(f'{SSH_CMD} "cd {REMOTE_DIR} && python evaluation/results_tracker.py --summary 2>/dev/null || echo \'No results yet\'"')

def download_results():
    """Download results from GPU server."""
    print("\n" + "="*50)
    print("  DOWNLOADING RESULTS")
    print("="*50)

    local_dir = Path(__file__).parent
    results_dir = local_dir / "evaluation" / "results"
    results_dir.mkdir(exist_ok=True)

    files = [
        "evaluation/comparison_results.json",
        "evaluation/results_template.csv",
        "evaluation/results_summary.csv",
        "evaluation/results_data.json",
        "experiment_output.log",
    ]

    for f in files:
        remote_path = f"{GPU_USER}@{GPU_HOST}:{REMOTE_DIR}/{f}"
        local_path = results_dir / Path(f).name
        run_cmd(f'{SCP_CMD} {remote_path} "{local_path}"', check=False)

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deploy experiments to GPU server")
    parser.add_argument('--upload', action='store_true', help="Upload files to GPU")
    parser.add_argument('--install', action='store_true', help="Install dependencies")
    parser.add_argument('--run-comparison', action='store_true', help="Run before/after comparison")
    parser.add_argument('--run-experiments', action='store_true', help="Run full experiment suite")
    parser.add_argument('--status', action='store_true', help="Check experiment status")
    parser.add_argument('--download', action='store_true', help="Download results")
    parser.add_argument('--full-setup', action='store_true', help="Upload + install + run comparison")

    args = parser.parse_args()

    if not any(vars(args).values()):
        # Default: show help (don't display credentials)
        print("Deploy to GPU Server")
        print("="*50)
        if GPU_HOST:
            print(f"Host: {GPU_HOST}:{GPU_PORT}")
            print(f"Remote dir: {REMOTE_DIR}")
        else:
            print("Host: Not configured (set ANDRAEUS_GPU_HOST)")
        print("\nOptions:")
        print("  --upload          Upload experiment files")
        print("  --install         Install Python dependencies")
        print("  --run-comparison  Run before/after comparison")
        print("  --run-experiments Run full 36-experiment suite ([amount])")
        print("  --status          Check experiment progress")
        print("  --download        Download results")
        print("  --full-setup      Upload + install + comparison")
        return

    # Check connection first
    if not check_connection():
        print("\nCannot connect to GPU server!")
        print("Check your credentials and network connection.")
        print("Test manually: ssh -p $ANDRAEUS_GPU_PORT $ANDRAEUS_GPU_USER@$ANDRAEUS_GPU_HOST")
        return

    if args.full_setup:
        upload_files()
        install_dependencies()
        run_before_after_comparison()
    else:
        if args.upload:
            upload_files()
        if args.install:
            install_dependencies()
        if args.run_comparison:
            run_before_after_comparison()
        if args.run_experiments:
            run_full_experiments()
        if args.status:
            check_status()
        if args.download:
            download_results()

if __name__ == "__main__":
    main()
