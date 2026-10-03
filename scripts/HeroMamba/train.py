"""
Hero-Mamba Training Pipeline.
Paper: arXiv:2404.16266 - High-Efficiency and Robust Underwater Image Restoration
with State Space Models and Physical Priors.

Reuses core data loaders, metrics, and TeeLogger from scripts.waterMamba.core.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.HeroMamba.heromamba_model import HeroMamba, HeroLoss, HAS_MAMBA
from scripts.waterMamba.core import (
    setup_tee_logger,
    calc_psnr,
    calc_ssim,
    get_dataloader,
)


def train_epoch(net, dataloader, optimizer, criterion, device, epoch, total_epochs, log_interval=10):
    net.train()
    running_loss = 0.0
    num_batches = len(dataloader)
    start_time = time.time()

    for batch_idx, (raw, ref, _) in enumerate(dataloader):
        raw = raw.to(device)
        ref = ref.to(device)

        optimizer.zero_grad()
        enhanced = net(raw)
        loss = criterion(enhanced, ref, raw)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            avg_loss = running_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(
                f"[Epoch {epoch:03d}/{total_epochs:03d}] "
                f"Batch {batch_idx + 1:04d}/{num_batches:04d} | "
                f"HeroLoss: {loss.item():.5f} (Avg: {avg_loss:.5f}) | "
                f"Speed: {(batch_idx + 1) / elapsed:.2f} batch/s"
            )

    return running_loss / max(1, num_batches)


def evaluate(net, dataloader, criterion, device, max_eval_batches=None):
    net.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for i, (raw, ref, _) in enumerate(dataloader):
            if max_eval_batches and i >= max_eval_batches:
                break
            raw = raw.to(device)
            ref = ref.to(device)
            enhanced = net(raw).clamp(0.0, 1.0)

            loss = criterion(enhanced, ref, raw)
            total_loss += loss.item()
            total_psnr += calc_psnr(enhanced, ref)
            total_ssim += calc_ssim(enhanced, ref)
            count += 1

    count = max(1, count)
    return {
        "loss": total_loss / count,
        "psnr": total_psnr / count,
        "ssim": total_ssim / count,
        "samples_evaluated": count,
    }


def train(args):
    device_name = "cpu" if args.cpu else "cuda"
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    dims = (args.base_dim, args.base_dim * 2, args.base_dim * 4, args.base_dim * 8)
    net = HeroMamba(dims=dims, scan_mode=args.scan_mode).to(device)

    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"Loading checkpoint weights from: {args.resume_checkpoint}")
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        net.load_state_dict(state_dict)
        print("Model weights loaded successfully.")

    print("\n" + "=" * 80)
    print(" STARTING HERO-MAMBA TRAINING")
    print(f" Device:         {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f" Fast SSM Mamba: {HAS_MAMBA}")
    print(f" Scan Mode:      {args.scan_mode} (using 2D scan: {net.ss2d_rgb.use_2d})")
    print(f" Resolution:     {args.img_size}x{args.img_size}")
    print(f" Dims:           {dims}")
    print(f" Batch Size:     {args.batch_size}")
    print(f" Learning Rate:  {args.lr}")
    print(f" Epochs:         {args.epochs}")
    print("=" * 80)

    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f" Model parameters: {total_params:.3f} M")

    train_loader = get_dataloader(
        data_dir=args.data_dir,
        split="train",
        dataset_type=args.dataset_type,
        splits_csv=args.splits_csv,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    val_loader = get_dataloader(
        data_dir=args.data_dir,
        split="val",
        dataset_type=args.dataset_type,
        splits_csv=args.splits_csv,
        img_size=args.img_size,
        batch_size=1,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # HeroLoss with optional VGG contrastive term
    gamma = 0.0 if args.no_vgg else args.gamma
    criterion = HeroLoss(alpha=args.alpha, beta=args.beta, gamma=gamma, pretrained_vgg=not args.no_vgg).to(device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_psnr = -1.0
    best_ckpt_path = os.path.join(args.checkpoint_dir, "heromamba_best.pth")
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "heromamba_latest.pth")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(
            net=net,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
        )
        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        val_metrics = evaluate(net, val_loader, criterion, device, max_eval_batches=args.max_val_batches)
        epoch_time = time.time() - epoch_start

        print(
            f"--> [Epoch {epoch:03d}/{args.epochs:03d} Completed in {epoch_time:.1f}s] "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_metrics['loss']:.5f} | "
            f"Val PSNR: {val_metrics['psnr']:.2f} dB | "
            f"Val SSIM: {val_metrics['ssim']:.4f} | "
            f"LR: {curr_lr:.6f}"
        )

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_psnr": best_val_psnr,
                "val_metrics": val_metrics,
            },
            latest_ckpt_path,
        )

        # Save best checkpoint
        if val_metrics["psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["psnr"]
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "val_psnr": best_val_psnr,
                    "val_ssim": val_metrics["ssim"],
                },
                best_ckpt_path,
            )
            print(f"    *** NEW BEST PSNR: {best_val_psnr:.2f} dB -> Saved to {best_ckpt_path} ***")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            p_path = os.path.join(args.checkpoint_dir, f"heromamba_epoch_{epoch:03d}.pth")
            torch.save({"epoch": epoch, "state_dict": net.state_dict()}, p_path)
            print(f"    [Checkpoint Saved] -> {p_path}")

    print("\n Training Complete!")
    print(f" Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f" Best Checkpoint:     {os.path.abspath(best_ckpt_path)}")
    return best_ckpt_path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hero-Mamba Training Suite")

    parser.add_argument("--base_dim", type=int, default=32, help="Base channel dimension C (default: 32)")
    parser.add_argument("--img_size", type=int, default=256, help="Square input resolution (default: 256)")
    parser.add_argument(
        "--scan_mode",
        type=str,
        choices=["auto", "2d", "1d"],
        default="auto",
        help="Scan mode: 2d (fast spatial coordinate sweeps), 1d (flattened 4-way), auto",
    )

    parser.add_argument("--dataset_type", type=str, choices=["lsui", "folder"], default="lsui")
    parser.add_argument("--data_dir", type=str, default="dataset/LSUI", help="Dataset root directory")
    parser.add_argument("--splits_csv", type=str, default=None, help="Path to splits csv")

    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default: 0)")

    parser.add_argument("--alpha", type=float, default=0.3, help="L1 loss weight")
    parser.add_argument("--beta", type=float, default=0.8, help="SSIM loss weight")
    parser.add_argument("--gamma", type=float, default=0.1, help="Contrastive VGG loss weight")
    parser.add_argument("--no_vgg", action="store_true", help="Disable VGG contrastive loss (faster/offline)")

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/heromamba", help="Save directory")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for Tee run logs")
    parser.add_argument("--save_every", type=int, default=5, help="Checkpoint interval")
    parser.add_argument("--log_interval", type=int, default=10, help="Batch log interval")

    parser.add_argument("--max_samples", type=int, default=None, help="Cap dataset samples for quick testing")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Cap validation batches")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")

    return parser.parse_args()


def main():
    args = parse_arguments()
    log_path = setup_tee_logger(log_dir=args.log_dir, run_name="heromamba_train")
    print(f"Command line arguments: {vars(args)}")

    try:
        train(args)
    finally:
        print(f"\n[Done] Training finished. Full logs saved at: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
