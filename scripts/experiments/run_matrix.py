"""
Master Orchestrator and Experiment Matrix Runner.
Features:
1. Pause & Resume Capability: Automatically detects completed runs in experiment_matrix.csv and skips them.
2. Group or Individual Execution: Run a single model (--run W-1a), a whole group (--group A), or all runs.
3. Clean CLI reporting: View status of all runs at a glance (--list).
4. Safe execution: Automatically passes memory guardrails, logging, and metrics recording.
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Master Registry of All Experiments from the HTML Specification + Adaptive Gate Extension
EXPERIMENT_REGISTRY = [
    # Group A: Attention Variations
    {
        "run_id": "W-1a", "group": "A", "model": "wm", "keep_ccoss": 0, "attn": "sca",
        "act": "base", "norm": "base", "scales": 0, "scan": "pooled_ssm", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "WaterMamba: Replace CCOSS with Simplified Channel Attention (SCA)"
    },
    {
        "run_id": "W-1b", "group": "A", "model": "wm", "keep_ccoss": 0, "attn": "rc",
        "act": "base", "norm": "base", "scales": 0, "scan": "pooled_ssm", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "WaterMamba: Replace CCOSS with Row/Column Spatial Gate"
    },
    {
        "run_id": "W-1c", "group": "A", "model": "wm", "keep_ccoss": 0, "attn": "rc_sca",
        "act": "base", "norm": "base", "scales": 0, "scan": "pooled_ssm", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "WaterMamba: Replace CCOSS with Spatial then Channel Attention"
    },
    {
        "run_id": "H-1a", "group": "A", "model": "hm", "keep_ccoss": 0, "attn": "sca",
        "act": "base", "norm": "bn", "scales": 0, "scan": "pooled_ssm", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "Hero-Mamba: Add SCA channel attention after spatial scan"
    },
    {
        "run_id": "H-1b", "group": "A", "model": "hm", "keep_ccoss": 0, "attn": "rc",
        "act": "base", "norm": "bn", "scales": 0, "scan": "pooled_ssm", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "Hero-Mamba: Add Row/Column spatial gate after spatial scan"
    },
    {
        "run_id": "H-1c", "group": "A", "model": "hm", "keep_ccoss": 0, "attn": "rc_sca",
        "act": "base", "norm": "bn", "scales": 0, "scan": "pooled_ssm", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "Hero-Mamba: Add Row/Column spatial gate then SCA"
    },

    # Group B: NAFNet & Gating Variations (Simple Gate + Adaptive Gate)
    {
        "run_id": "W-N1-SG", "group": "B", "model": "wm", "keep_ccoss": 1, "attn": "none",
        "act": "simplegate", "norm": "base", "scales": 0, "scan": "pooled_ssm", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "WaterMamba: Simple Gate instead of activation in MSFFN"
    },
    {
        "run_id": "W-N1-Ada", "group": "B", "model": "wm", "keep_ccoss": 1, "attn": "none",
        "act": "adaptivegate", "norm": "base", "scales": 0, "scan": "pooled_ssm", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "WaterMamba: Adaptive Gate (AdaGate) instead of activation in MSFFN"
    },
    {
        "run_id": "H-N1-SG", "group": "B", "model": "hm", "keep_ccoss": 0, "attn": "none",
        "act": "simplegate", "norm": "bn", "scales": 0, "scan": "pooled_ssm", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "Hero-Mamba: Simple Gate in MS-Fusion branches"
    },
    {
        "run_id": "H-N1-Ada", "group": "B", "model": "hm", "keep_ccoss": 0, "attn": "none",
        "act": "adaptivegate", "norm": "bn", "scales": 0, "scan": "pooled_ssm", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "Hero-Mamba: Adaptive Gate (AdaGate) in MS-Fusion branches"
    },
    {
        "run_id": "H-D2", "group": "B", "model": "hm", "keep_ccoss": 0, "attn": "none",
        "act": "base", "norm": "ln", "scales": 0, "scan": "pooled_ssm", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "Hero-Mamba: LayerNorm instead of BatchNorm in Decoder"
    },
    {
        "run_id": "H-D3-SG", "group": "B", "model": "hm", "keep_ccoss": 0, "attn": "none",
        "act": "simplegate", "norm": "ln", "scales": 1, "scan": "pooled_ssm", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "Hero-Mamba: LayerNorm + SimpleGate + learnable beta,gamma scales in Decoder"
    },
    {
        "run_id": "H-D3-Ada", "group": "B", "model": "hm", "keep_ccoss": 0, "attn": "none",
        "act": "adaptivegate", "norm": "ln", "scales": 1, "scan": "pooled_ssm", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "Hero-Mamba: LayerNorm + AdaptiveGate + learnable beta,gamma scales in Decoder"
    },

    # Group D: Structural Ablations
    {
        "run_id": "W-R2", "group": "D", "model": "wm", "keep_ccoss": 1, "attn": "none",
        "act": "base", "norm": "base", "scales": 0, "scan": "pooled_conv", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "WaterMamba: Ablation with 1D Conv replacing S6 scan"
    },
    {
        "run_id": "H-R2", "group": "D", "model": "hm", "keep_ccoss": 0, "attn": "none",
        "act": "base", "norm": "bn", "scales": 0, "scan": "pooled_conv", "res": 128, "batch_size": 4, "epochs": 5,
        "desc": "Hero-Mamba: Ablation with 1D Conv replacing S6 scan"
    },
    {
        "run_id": "W-R0", "group": "D", "model": "wm", "keep_ccoss": 1, "attn": "none",
        "act": "base", "norm": "base", "scales": 0, "scan": "full", "res": 64, "batch_size": 1, "epochs": 5,
        "desc": "WaterMamba: Full unrolled 4-way scan on 64x64 crops"
    },
    {
        "run_id": "W-R1", "group": "D", "model": "wm", "keep_ccoss": 1, "attn": "none",
        "act": "base", "norm": "base", "scales": 0, "scan": "pooled_ssm", "res": 64, "batch_size": 1, "epochs": 5,
        "desc": "WaterMamba: Pooled scan on 64x64 crops"
    },
    {
        "run_id": "H-R0", "group": "D", "model": "hm", "keep_ccoss": 0, "attn": "none",
        "act": "base", "norm": "bn", "scales": 0, "scan": "full", "res": 64, "batch_size": 1, "epochs": 5,
        "desc": "Hero-Mamba: Full unrolled 4-way scan on 64x64 crops"
    },
    {
        "run_id": "H-R1", "group": "D", "model": "hm", "keep_ccoss": 0, "attn": "none",
        "act": "base", "norm": "bn", "scales": 0, "scan": "pooled_ssm", "res": 64, "batch_size": 1, "epochs": 5,
        "desc": "Hero-Mamba: Pooled scan on 64x64 crops"
    },
]


def get_completed_runs(matrix_csv_path):
    if not os.path.exists(matrix_csv_path):
        return set()
    try:
        df = pd.read_csv(matrix_csv_path)
        if "run_id" in df.columns and "status" in df.columns:
            completed = set(df[df["status"] == "COMPLETED"]["run_id"].values)
            return completed
    except Exception:
        pass
    return set()


def print_status_table(matrix_csv_path):
    print("\n" + "=" * 105)
    print(" EXPERIMENT MATRIX STATUS")
    print("=" * 105)
    completed_runs = get_completed_runs(matrix_csv_path)
    df_results = pd.read_csv(matrix_csv_path) if os.path.exists(matrix_csv_path) else pd.DataFrame()

    print(f"{'Run ID':<10} | {'Group':<5} | {'Model':<5} | {'Scan':<10} | {'Attn':<8} | {'Act':<12} | {'Status':<12} | {'Best PSNR':<9} | {'SSIM':<7}")
    print("-" * 105)
    for exp in EXPERIMENT_REGISTRY:
        rid = exp["run_id"]
        status = "COMPLETED" if rid in completed_runs else "PENDING"
        psnr_str, ssim_str = "—", "—"

        if not df_results.empty and rid in df_results["run_id"].values:
            row = df_results[df_results["run_id"] == rid].iloc[-1]
            psnr_val = row.get("best_val_psnr", None)
            ssim_val = row.get("best_val_ssim", None)
            if pd.notna(psnr_val):
                psnr_str = f"{psnr_val:.2f} dB"
            if pd.notna(ssim_val):
                ssim_str = f"{ssim_val:.4f}"

        print(f"{rid:<10} | {exp['group']:<5} | {exp['model']:<5} | {exp['scan']:<10} | {exp['attn']:<8} | {exp['act']:<12} | {status:<12} | {psnr_str:<9} | {ssim_str:<7}")
    print("=" * 105 + "\n")


def execute_run(exp, python_bin, matrix_csv_path):
    cmd = [
        python_bin, "scripts/experiments/train_experiment.py",
        "--run_id", exp["run_id"],
        "--group", exp["group"],
        "--model", exp["model"],
        "--keep_ccoss", str(exp["keep_ccoss"]),
        "--attn", exp["attn"],
        "--act", exp["act"],
        "--norm", exp["norm"],
        "--scales", str(exp["scales"]),
        "--scan", exp["scan"],
        "--res", str(exp["res"]),
        "--batch_size", str(exp["batch_size"]),
        "--epochs", str(exp["epochs"]),
        "--subset", "0.25",
        "--matrix_csv", matrix_csv_path
    ]

    print("\n" + "#" * 80)
    print(f" LAUNCHING: {exp['run_id']} - {exp['desc']}")
    print(f" Command: {' '.join(cmd)}")
    print("#" * 80 + "\n")

    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n[WARNING]: Run {exp['run_id']} exited with return code {res.returncode}")
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Master Post-Baseline Matrix Runner with Pause & Resume")
    parser.add_argument("--run", type=str, default=None, help="Run a single specific experiment (e.g. W-1a, H-N1-SG)")
    parser.add_argument("--group", type=str, default=None, choices=["A", "B", "D", "all"], help="Run all experiments in a specific group")
    parser.add_argument("--list", action="store_true", help="Print experiment status matrix and exit")
    parser.add_argument("--force", action="store_true", help="Force rerun even if already marked COMPLETED")
    parser.add_argument("--dry_run", action="store_true", help="Print planned commands without executing")
    parser.add_argument("--matrix_csv", type=str, default="logs/experiments/experiment_matrix.csv", help="Path to tracking CSV")
    args = parser.parse_args()

    matrix_csv_path = args.matrix_csv

    if args.list:
        print_status_table(matrix_csv_path)
        return

    completed_runs = get_completed_runs(matrix_csv_path) if not args.force else set()
    python_bin = sys.executable

    # Filter target runs
    targets = []
    if args.run:
        matched = [e for e in EXPERIMENT_REGISTRY if e["run_id"].lower() == args.run.lower()]
        if not matched:
            print(f"Error: Run '{args.run}' not found in registry. Use --list to see all available runs.")
            return
        targets = matched
    elif args.group and args.group.lower() != "all":
        targets = [e for e in EXPERIMENT_REGISTRY if e["group"].upper() == args.group.upper()]
    else:
        targets = EXPERIMENT_REGISTRY

    # Filter out already completed runs
    queue = [e for e in targets if e["run_id"] not in completed_runs]
    skipped = [e for e in targets if e["run_id"] in completed_runs]

    print(f"\n[Matrix Runner Plan] Total in Scope: {len(targets)} | Already Completed: {len(skipped)} | Queued to Run: {len(queue)}")
    if skipped:
        print(f"Skipping completed: {', '.join([s['run_id'] for s in skipped])}")

    if args.dry_run:
        print("\nDry run requested. Planned execution order:")
        for idx, exp in enumerate(queue, 1):
            print(f" {idx:02d}. {exp['run_id']} (Group {exp['group']}): {exp['desc']}")
        return

    if not queue:
        print("\nAll targeted experiments have already COMPLETED! Use --force to re-run or check with --list.")
        print_status_table(matrix_csv_path)
        return

    # Execute queue with pause & resume support
    for idx, exp in enumerate(queue, 1):
        print(f"\n[{idx}/{len(queue)}] Starting {exp['run_id']}...")
        success = execute_run(exp, python_bin, matrix_csv_path)
        if not success:
            print(f"[PAUSED/HALTED] Experiment {exp['run_id']} encountered an error. You can fix the issue and resume seamlessly.")
            break

    print_status_table(matrix_csv_path)


if __name__ == "__main__":
    main()
