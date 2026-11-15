"""
Master 30-Hour Dual-Model Orchestrator for Exact Research Paper Training.
Trains BOTH WaterMamba (arXiv:2405.08419) and Hero-Mamba (arXiv:2404.16266)
with zero architectural changes (exact 1D 4-directional flattened selective scan)
within a strict 30-hour total wall-clock budget on RTX 4060 Laptop GPU.

Usage:
  python scripts/paper_exact/run_both_30h.py
  python scripts/paper_exact/run_both_30h.py --resume
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)


def run_stage(model_code, max_hours, args):
    label = "watermamba_paper" if model_code == "wm" else "heromamba_paper"
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "paper_exact" / label
    test_json = ckpt_dir / "test_results.json"

    if args.resume and test_json.exists():
        print(f"[Orchestrator] Skipping {label} (already completed & evaluated: {test_json})")
        return True

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "paper_exact" / "train_paper_model.py"),
        "--model", model_code,
        "--max_hours", f"{max_hours:.4f}",
    ]
    if model_code == "wm":
        cmd += ["--batch_size", str(args.wm_batch_size), "--grad_accum", str(args.wm_grad_accum)]
        if args.wm_epochs is not None:
            cmd += ["--epochs", str(args.wm_epochs)]
    else:
        cmd += ["--batch_size", str(args.hm_batch_size), "--grad_accum", str(args.hm_grad_accum)]
        if args.hm_epochs is not None:
            cmd += ["--epochs", str(args.hm_epochs)]

    if args.resume:
        cmd.append("--resume")
    if args.no_amp:
        cmd.append("--no_amp")
    if args.no_vgg:
        cmd.append("--no_vgg")

    print("\n" + "#" * 84)
    print(f" LAUNCHING STAGE: {label.upper()} (Max Budget: {max_hours:.2f} hours)")
    print(f" Command: {' '.join(cmd)}")
    print("#" * 84 + "\n")

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"\n[WARNING] Stage {label} exited with code {proc.returncode}. You can resume with --resume.")
        return False
    return True


def print_final_comparison():
    wm_json = PROJECT_ROOT / "checkpoints" / "paper_exact" / "watermamba_paper" / "test_results.json"
    hm_json = PROJECT_ROOT / "checkpoints" / "paper_exact" / "heromamba_paper" / "test_results.json"

    rows = [
        {"Variant": "WaterMamba (Paper-Exact 1D Scan)", "File": wm_json},
        {"Variant": "Hero-Mamba (Paper-Exact 1D Scan)", "File": hm_json},
    ]

    summary_rows = []
    print("\n" + "=" * 96)
    print(" FINAL PAPER-EXACT vs. 2D POOLED & EXPERIMENTAL ARCHITECTURES SUMMARY")
    print("=" * 96)
    print(f"{'Model / Variant':<36} | {'Scan Type':<14} | {'Test PSNR':<10} | {'PSNR_mm':<10} | {'Test SSIM':<10} | {'Hours'}")
    print("-" * 96)

    for item in rows:
        if item["File"].exists():
            with open(item["File"], "r", encoding="utf-8") as f:
                d = json.load(f)
            print(
                f"{item['Variant']:<36} | {'1D 4-Way Exact':<14} | "
                f"{d['test_psnr']:7.2f} dB | {d['test_psnr_mm']:7.2f} dB | "
                f"{d['test_ssim']:10.4f} | {d['total_hours']:.2f}h ({d['best_epoch']} ep)"
            )
            summary_rows.append(d)
        else:
            print(f"{item['Variant']:<36} | {'PENDING':<14} | {'—':<10} | {'—':<10} | {'—':<10} | —")

    print("-" * 96)
    print(f"{'WaterMamba (2D Pooled Baseline)':<36} | {'2D Row/Col Avg':<14} | {'26.77 dB':<10} | {'28.54 dB':<10} | {'0.9080':<10} | Full 10 ep")
    print(f"{'Hero-Mamba (2D Pooled Baseline)':<36} | {'2D Row/Col Avg':<14} | {'23.24 dB':<10} | {'25.10 dB':<10} | {'0.8940':<10} | Full 10 ep")
    print(f"{'W-1a (SCA replacing CCOSS, 25% sub)':<36} | {'2D Row/Col Avg':<14} | {'24.50* dB':<10} | {'26.32* dB':<10} | {'0.9080*':<10} | 5 ep (Val)")
    print(f"{'H-1b (RC Spatial Gate, 25% sub)':<36} | {'2D Row/Col Avg':<14} | {'22.67* dB':<10} | {'24.39* dB':<10} | {'0.8801*':<10} | 5 ep (Val)")
    print(f"{'H-D3-Ada (LN+AdaGate+Scales, 25%)':<36} | {'2D Row/Col Avg':<14} | {'22.46* dB':<10} | {'24.17* dB':<10} | {'0.8762*':<10} | 5 ep (Val)")
    print("=" * 96)

    if summary_rows:
        out_csv = PROJECT_ROOT / "logs" / "paper_exact" / "paper_exact_results.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSaved paper-exact benchmark results -> {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="30-Hour Exact Paper Dual-Model Orchestrator")
    parser.add_argument("--total_hours", type=float, default=30.0, help="Total combined hour budget (default: 30.0)")
    parser.add_argument("--wm_hours", type=float, default=15.0, help="Hours allocated to WaterMamba (default: 15.0)")
    parser.add_argument("--wm_epochs", type=int, default=None, help="Optional max epochs for WaterMamba (default: 750, auto-capped by --wm_hours)")
    parser.add_argument("--hm_epochs", type=int, default=None, help="Optional max epochs for Hero-Mamba (default: 250, auto-capped by time budget)")
    parser.add_argument("--wm_batch_size", type=int, default=2, help="WaterMamba batch size (paper default: 2)")
    parser.add_argument("--wm_grad_accum", type=int, default=1, help="WaterMamba grad accum (default: 1)")
    parser.add_argument("--hm_batch_size", type=int, default=4, help="Hero-Mamba batch size (paper default: 4)")
    parser.add_argument("--hm_grad_accum", type=int, default=1, help="Hero-Mamba grad accum (default: 1)")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted run from latest checkpoint")
    parser.add_argument("--no_amp", action="store_true", help="Disable BF16 autocast")
    parser.add_argument("--no_vgg", action="store_true", help="Disable VGG19 term in HeroLoss")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 84)
    print(" 30-HOUR EXACT RESEARCH PAPER TRAINING SUITE (RTX 4060 LAPTOP)")
    print(f" Total Budget: {args.total_hours:.2f} hours | WaterMamba: {args.wm_hours:.2f}h | Hero-Mamba: remaining")
    print("=" * 84)

    ok_wm = run_stage("wm", args.wm_hours, args)
    wm_json = PROJECT_ROOT / "checkpoints" / "paper_exact" / "watermamba_paper" / "test_results.json"
    wm_used_h = (time.time() - t0) / 3600.0
    if wm_json.exists():
        try:
            with open(wm_json, "r", encoding="utf-8") as f:
                wm_used_h = max(wm_used_h, float(json.load(f).get("total_hours", wm_used_h)))
        except Exception:
            pass
    hm_budget = max(1.0, args.total_hours - wm_used_h)

    if ok_wm:
        run_stage("hm", hm_budget, args)
    else:
        print("\n[Orchestrator] WaterMamba did not complete cleanly. Fix/resume before starting Hero-Mamba.")

    print_final_comparison()


if __name__ == "__main__":
    main()
