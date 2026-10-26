"""
Master Training Script for Single Post-Baseline Experiments.
Executes any experiment from the matrix with:
1. Dynamic model instantiation via models_factory.
2. Automatic ExperimentTeeLogger (console + disk log file).
3. Step-1 GPU VRAM check with 7.0 GB safety threshold.
4. Comprehensive validation: Loss, PSNR, SSIM, Mean-Matched PSNR.
5. Auto-appending results into logs/experiments/experiment_matrix.csv.
"""

import os
import sys
import time
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.waterMamba.core import LSUIPairDataset, get_dataloader
from scripts.experiments.models_factory import build_experiment_model
from scripts.experiments.metrics_tracker import (
    calc_psnr,
    calc_ssim,
    calc_mean_matched_psnr,
    ExperimentTeeLogger,
    check_vram_guard,
    record_experiment_result,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Post-Baseline Single Experiment Runner")
    # Experiment Identifiers
    parser.add_argument("--run_id", type=str, required=True, help="Unique run identifier (e.g., W-1a, H-N1-SG)")
    parser.add_argument("--group", type=str, default="A", help="Experiment group: A, B, C, or D")
    parser.add_argument("--model", type=str, required=True, choices=["wm", "hm", "watermamba", "heromamba"], help="Model backbone")

    # Modular Architectural Flags
    parser.add_argument("--keep_ccoss", type=int, default=1, choices=[0, 1], help="Keep original CCOSS in WaterMamba")
    parser.add_argument("--attn", type=str, default="none", choices=["none", "sca", "rc", "rc_sca"], help="Attention type")
    parser.add_argument("--act", type=str, default="base", choices=["base", "simplegate", "adaptivegate"], help="Activation / Gating type")
    parser.add_argument("--norm", type=str, default="base", choices=["base", "bn", "ln"], help="Decoder normalization (hm only)")
    parser.add_argument("--scales", type=int, default=0, choices=[0, 1], help="Learnable residual scales in decoder (hm only)")
    parser.add_argument("--scan", type=str, default="pooled_ssm", choices=["pooled_ssm", "pooled_conv", "full"], help="Spatial scan type")

    # Fixed Budget & Training Settings
    parser.add_argument("--res", type=int, default=128, help="Crop/resize image resolution (128 for standard, 64 for crop runs)")
    parser.add_argument("--subset", type=float, default=0.25, help="Dataset subset fraction (e.g. 0.25 for 25% subset)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (4 for standard, 1 for crop runs)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--max_vram_gb", type=float, default=7.0, help="VRAM safety threshold (GB)")

    # Data & Paths
    parser.add_argument("--data_dir", type=str, default="dataset/LSUI", help="Dataset root directory")
    parser.add_argument("--splits_csv", type=str, default="dataset/LSUI/lsui_splits.csv", help="Splits CSV file")
    parser.add_argument("--log_dir", type=str, default="logs/experiments", help="Directory for log files")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/experiments", help="Directory for model checkpoints")
    parser.add_argument("--matrix_csv", type=str, default="logs/experiments/experiment_matrix.csv", help="Summary CSV tracking file")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def evaluate(net, dataloader, device):
    net.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_psnr_mm = 0.0
    count = 0

    with torch.no_grad():
        for raw, ref, _ in dataloader:
            raw = raw.to(device)
            ref = ref.to(device)
            enhanced = net(raw).clamp(0.0, 1.0)

            total_psnr += calc_psnr(enhanced, ref)
            total_ssim += calc_ssim(enhanced, ref)
            total_psnr_mm += calc_mean_matched_psnr(enhanced, ref)
            count += 1

    count = max(1, count)
    return {
        "psnr": total_psnr / count,
        "ssim": total_ssim / count,
        "psnr_mm": total_psnr_mm / count,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. Setup TeeLogger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"{args.run_id}_{timestamp}.log")
    tee_logger = ExperimentTeeLogger(log_file)

    print("=" * 80)
    print(f" EXPERIMENT RUNNER: {args.run_id} [Group {args.group}]")
    print(f" Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Log File:   {log_file}")
    print(f" Model:      {args.model} | Scan: {args.scan} | Attn: {args.attn} | Act: {args.act}")
    print(f" Resolution: {args.res}x{args.res} | Subset: {args.subset*100:.0f}% | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 2. Build Model
    norm_val = "bn" if args.norm == "base" else args.norm
    net = build_experiment_model(
        model_type=args.model,
        keep_ccoss=args.keep_ccoss,
        attn=args.attn,
        act=args.act,
        norm=norm_val,
        scales=args.scales,
        scan=args.scan,
        img_size=args.res
    ).to(device)

    total_params_M = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"Model Parameters: {total_params_M:.3f} M")

    # 3. Dataloaders with subset filtering
    total_train_pairs = 3423
    max_train_samples = int(total_train_pairs * args.subset) if args.subset < 1.0 else None

    train_loader = get_dataloader(
        data_dir=args.data_dir,
        split="train",
        dataset_type="lsui",
        splits_csv=args.splits_csv,
        img_size=args.res,
        batch_size=args.batch_size,
        num_workers=0,
        max_samples=max_train_samples
    )
    val_loader = get_dataloader(
        data_dir=args.data_dir,
        split="val",
        dataset_type="lsui",
        splits_csv=args.splits_csv,
        img_size=args.res,
        batch_size=1,
        num_workers=0,
        max_samples=100  # fast validation subset
    )

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.L1Loss()

    ckpt_dir = os.path.join(args.checkpoint_dir, args.run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_psnr = -1.0
    best_val_ssim = -1.0
    best_val_psnr_mm = -1.0
    peak_vram_mb = 0.0
    epoch_times = []
    final_train_loss = 0.0

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            net.train()
            running_loss = 0.0
            num_batches = len(train_loader)

            for step, (raw, ref, _) in enumerate(train_loader):
                raw = raw.to(device)
                ref = ref.to(device)

                optimizer.zero_grad()
                enhanced = net(raw)
                loss = criterion(enhanced, ref)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Step 1 VRAM Guard Check
                if epoch == 1 and step == 0:
                    peak_vram_mb = check_vram_guard(step=1, max_vram_gb=args.max_vram_gb)

            scheduler.step()
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            final_train_loss = running_loss / max(1, num_batches)

            # Evaluate on validation set
            val_res = evaluate(net, val_loader, device)
            if torch.cuda.is_available():
                peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / (1024 ** 2))

            print(
                f"--> [Epoch {epoch:02d}/{args.epochs:02d} - {epoch_time:.1f}s] "
                f"Train Loss: {final_train_loss:.5f} | "
                f"Val PSNR: {val_res['psnr']:.2f} dB | "
                f"Val SSIM: {val_res['ssim']:.4f} | "
                f"Val PSNR-MM: {val_res['psnr_mm']:.2f} dB | "
                f"Peak VRAM: {peak_vram_mb:.1f} MB"
            )

            # Track best metrics
            if val_res["psnr"] > best_val_psnr:
                best_val_psnr = val_res["psnr"]
                best_val_ssim = val_res["ssim"]
                best_val_psnr_mm = val_res["psnr_mm"]
                torch.save(net.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))

        status = "COMPLETED"
    except Exception as e:
        status = f"FAILED: {str(e)[:100]}"
        print(f"\n[ERROR in {args.run_id}]: {e}")
        raise e
    finally:
        avg_sec = sum(epoch_times) / max(1, len(epoch_times))
        record = {
            "run_id": args.run_id,
            "group": args.group,
            "model": args.model,
            "keep_ccoss": args.keep_ccoss,
            "attn": args.attn,
            "act": args.act,
            "norm": args.norm,
            "scales": args.scales,
            "scan": args.scan,
            "res": args.res,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "params_M": round(total_params_M, 3),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "best_val_psnr": round(best_val_psnr, 2),
            "best_val_ssim": round(best_val_ssim, 4),
            "best_val_psnr_mm": round(best_val_psnr_mm, 2),
            "final_train_loss": round(final_train_loss, 5),
            "avg_sec_per_epoch": round(avg_sec, 1),
            "status": status,
        }
        record_experiment_result(args.matrix_csv, record)
        print("=" * 80)
        print(f" FINISHED: {args.run_id} | Status: {status} | Best PSNR: {best_val_psnr:.2f} dB | Best SSIM: {best_val_ssim:.4f}")
        print("=" * 80)
        tee_logger.close()


if __name__ == "__main__":
    main()
