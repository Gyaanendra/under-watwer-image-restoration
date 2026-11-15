"""
Exact Research Paper Trainer for WaterMamba (arXiv:2405.08419) & Hero-Mamba (arXiv:2404.16266).
Runs the unmodified 1D 4-directional flattened selective scan (scan_mode='1d')
accelerated by our native CUDA S6 kernel on the full LSUI dataset (256x256).

Features:
- 100% paper-exact network architectures (zero architectural alterations).
- RTX 4060 8GB VRAM safety: Step-1 VRAM monitor + optional BF16 AMP + gradient accumulation.
- Strict wall-clock time budgeting (--max_hours) with automatic CosineAnnealing T_max adaptation.
- Seamless pause & resume (--resume) across epochs with full optimizer/scheduler/time persistence.
- Automatic 429-image LSUI test evaluation at completion + Tee logging to disk.
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from scripts.paper_exact.cuda_ssm import install_fast_selective_scan
from scripts.waterMamba.watermamba_model import WaterMamba
from scripts.HeroMamba.heromamba_model import HeroMamba, HeroLoss
from scripts.waterMamba.core import (
    setup_tee_logger,
    calc_psnr,
    calc_ssim,
    get_dataloader,
)


def calc_mean_matched_psnr(pred, gt, max_val=1.0):
    """Mean-matched PSNR evaluating structural fidelity independent of global brightness shift."""
    pred_m = pred - pred.mean(dim=(2, 3), keepdim=True)
    gt_m = gt - gt.mean(dim=(2, 3), keepdim=True)
    mse = torch.mean((pred_m - gt_m) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(max_val / math.sqrt(mse.item()))


def build_paper_model_and_loss(args, device):
    """Instantiate exact paper architecture and loss function without alterations."""
    if args.model == "wm":
        net = WaterMamba(dim=32, img_size=args.img_size, blocks=1, scan_mode="1d").to(device)
        criterion = nn.L1Loss().to(device)
        is_hero = False
        paper_ref = "WaterMamba (arXiv:2405.08419) | 1D 4-Way SOSS + CCOSS + MSFFN + L1"
    elif args.model == "hm":
        net = HeroMamba(dims=(32, 64, 128, 256), scan_mode="1d").to(device)
        try:
            criterion = HeroLoss(alpha=0.3, beta=0.8, gamma=0.1, pretrained_vgg=not args.no_vgg).to(device)
        except Exception as exc:
            print(f"[WARNING] Pretrained VGG19 weights unavailable ({exc}); falling back to gamma=0 L1+SSIM.")
            criterion = HeroLoss(alpha=0.3, beta=0.8, gamma=0.0, pretrained_vgg=False).to(device)
        is_hero = True
        paper_ref = "Hero-Mamba (arXiv:2404.16266) | 1D 4-Way SS2D + FFT + Prior + HeroLoss"
    else:
        raise ValueError(f"Unsupported model '{args.model}'. Choose 'wm' or 'hm'.")
    return net, criterion, is_hero, paper_ref


def train_one_epoch(net, loader, optimizer, criterion, device, epoch, total_epochs,
                    is_hero=False, amp=True, grad_accum=1, log_interval=50,
                    check_vram=False, max_vram_gb=7.4):
    net.train()
    running_loss = 0.0
    num_batches = len(loader)
    t_start = time.time()

    optimizer.zero_grad(set_to_none=True)
    amp_dtype = torch.bfloat16 if (amp and device.type == "cuda") else torch.float32

    for batch_idx, (raw, ref, _) in enumerate(loader):
        raw = raw.to(device, non_blocking=True)
        ref = ref.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp and device.type == "cuda")):
            enhanced = net(raw)

        # Compute loss strictly in FP32 to prevent BF16 precision cancellation in SSIM (C1=1e-4, C2=9e-4)
        enh_f = enhanced.float()
        loss_raw = criterion(enh_f, ref.float(), raw.float()) if is_hero else criterion(enh_f, ref.float())
        loss = loss_raw / grad_accum

        loss.backward()

        if ((batch_idx + 1) % grad_accum == 0) or ((batch_idx + 1) == num_batches):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if check_vram and batch_idx == 0 and device.type == "cuda":
            torch.cuda.synchronize()
            peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(f"[VRAM Check @ Step 1] Peak Memory Allocated: {peak_gb * 1024:.1f} MB ({peak_gb:.2f} GB)")
            if peak_gb > max_vram_gb:
                raise MemoryError(
                    f"Peak VRAM {peak_gb:.2f} GB exceeded safety threshold {max_vram_gb:.2f} GB! "
                    f"Lower --batch_size (e.g. 1) and increase --grad_accum."
                )

        running_loss += loss_raw.item()

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            avg_loss = running_loss / (batch_idx + 1)
            elapsed = max(1e-6, time.time() - t_start)
            print(
                f"  [Epoch {epoch:03d}/{total_epochs:03d}] "
                f"Batch {batch_idx + 1:04d}/{num_batches:04d} | "
                f"Loss: {loss_raw.item():.5f} (Avg: {avg_loss:.5f}) | "
                f"Speed: {(batch_idx + 1) / elapsed:.2f} batch/s"
            )

    return running_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(net, loader, criterion, device, is_hero=False, amp=True):
    net.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_psnr_mm = 0.0
    total_ssim = 0.0
    total_lat_ms = 0.0
    count = 0

    amp_dtype = torch.bfloat16 if (amp and device.type == "cuda") else torch.float32

    for raw, ref, _ in loader:
        raw = raw.to(device, non_blocking=True)
        ref = ref.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(amp and device.type == "cuda")):
            enhanced = net(raw).clamp(0.0, 1.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_lat_ms += (time.time() - t0) * 1000.0

        enh_f = enhanced.float()
        ref_f = ref.float()
        raw_f = raw.float()
        loss = criterion(enh_f, ref_f, raw_f) if is_hero else criterion(enh_f, ref_f)

        total_loss += loss.item()
        total_psnr += calc_psnr(enh_f, ref_f)
        total_psnr_mm += calc_mean_matched_psnr(enh_f, ref_f)
        total_ssim += calc_ssim(enh_f, ref_f)
        count += 1

    count = max(1, count)
    return {
        "loss": total_loss / count,
        "psnr": total_psnr / count,
        "psnr_mm": total_psnr_mm / count,
        "ssim": total_ssim / count,
        "latency_ms": total_lat_ms / count,
        "samples_evaluated": count,
    }


def main():
    parser = argparse.ArgumentParser(description="Exact Research Paper Training Runner")
    parser.add_argument("--model", type=str, required=True, choices=["wm", "hm"],
                        help="'wm' = WaterMamba (2405.08419), 'hm' = Hero-Mamba (2404.16266)")
    parser.add_argument("--img_size", type=int, default=256, help="Paper image resolution (default: 256)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Max epochs (default: 750 for wm, 250 for hm)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Paper batch size (default: 2 for wm, 4 for hm)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial AdamW LR (default: 1e-4)")
    parser.add_argument("--max_hours", type=float, default=15.0,
                        help="Wall-clock time cap in hours for this model (default: 15.0)")
    parser.add_argument("--auto_tmax", action="store_true", default=True,
                        help="Adapt CosineAnnealingLR T_max to fit within --max_hours after Epoch 1")
    parser.add_argument("--no_amp", action="store_true", help="Disable BF16 mixed precision")
    parser.add_argument("--no_vgg", action="store_true", help="Disable VGG19 term in HeroLoss")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes (0 safest on Windows)")
    parser.add_argument("--log_interval", type=int, default=100, help="Batch print interval")
    parser.add_argument("--save_every", type=int, default=10, help="Periodic epoch checkpoint interval")
    parser.add_argument("--max_vram_gb", type=float, default=7.4, help="Step-1 VRAM safety threshold (GB)")
    parser.add_argument("--data_dir", type=str, default="dataset/LSUI", help="LSUI dataset path")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="logs/paper_exact", help="Tee log directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest.pth if present")
    parser.add_argument("--skip_cuda_verify", action="store_true", help="Skip CUDA vs PyTorch self-check")
    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = 750 if args.model == "wm" else 250
    if args.batch_size is None:
        args.batch_size = 2 if args.model == "wm" else 4

    model_label = "watermamba_paper" if args.model == "wm" else "heromamba_paper"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/paper_exact/{model_label}"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    log_path = setup_tee_logger(log_dir=args.log_dir, run_name=model_label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 84)
    print(f" EXACT PAPER RUNNER: {model_label.upper()}")
    print(f" Device:       {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f" Resolution:   {args.img_size}x{args.img_size} | Scan Mode: 1d (Exact Paper 4-Way)")
    print(f" Batch Size:   {args.batch_size} x GradAccum {args.grad_accum} = Effective Batch {args.batch_size * args.grad_accum}")
    print(f" Mixed Prec:   {'BF16 Autocast (S6 in FP32)' if not args.no_amp else 'Pure FP32'}")
    print(f" Time Budget:  {args.max_hours:.2f} hours | Target Max Epochs: {args.epochs}")
    print("=" * 84)

    # Compile/load native CUDA selective scan & verify exactness
    if device.type == "cuda":
        install_fast_selective_scan(verify=not args.skip_cuda_verify)

    net, criterion, is_hero, paper_ref = build_paper_model_and_loss(args, device)
    params_m = sum(p.numel() for p in net.parameters()) / 1e6
    print(f" Architecture: {paper_ref}")
    print(f" Parameters:   {params_m:.3f} M")

    train_loader = get_dataloader(
        data_dir=args.data_dir, split="train", dataset_type="lsui",
        img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_loader = get_dataloader(
        data_dir=args.data_dir, split="val", dataset_type="lsui",
        img_size=args.img_size, batch_size=1, num_workers=args.num_workers
    )
    test_loader = get_dataloader(
        data_dir=args.data_dir, split="test", dataset_type="lsui",
        img_size=args.img_size, batch_size=1, num_workers=args.num_workers
    )

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    latest_path = os.path.join(args.checkpoint_dir, "latest.pth")
    best_path = os.path.join(args.checkpoint_dir, "best.pth")
    results_json = os.path.join(args.checkpoint_dir, "test_results.json")

    start_epoch = 1
    best_val_psnr = -1.0
    best_val_ssim = 0.0
    prior_elapsed_s = 0.0
    tmax_adapted = False

    if args.resume and os.path.exists(latest_path):
        print(f"[Resume] Loading checkpoint from {latest_path} ...")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_psnr = ckpt.get("best_val_psnr", -1.0)
        best_val_ssim = ckpt.get("best_val_ssim", 0.0)
        prior_elapsed_s = ckpt.get("elapsed_s", 0.0)
        tmax_adapted = ckpt.get("tmax_adapted", False)
        print(
            f"[Resume] Resuming from Epoch {start_epoch} | "
            f"Best PSNR so far: {best_val_psnr:.2f} dB | "
            f"Prior Elapsed: {prior_elapsed_s / 3600:.2f}h"
        )

    session_start = time.time()
    max_budget_s = args.max_hours * 3600.0
    epoch_times = []

    for epoch in range(start_epoch, args.epochs + 1):
        total_elapsed_s = prior_elapsed_s + (time.time() - session_start)
        if total_elapsed_s >= max_budget_s:
            print(f"\n[Time Budget Reached] Elapsed {total_elapsed_s / 3600:.2f}h >= {args.max_hours:.2f}h. Stopping cleanly.")
            break

        if epoch_times:
            avg_ep_s = sum(epoch_times) / len(epoch_times)
            if total_elapsed_s + avg_ep_s * 1.03 > max_budget_s:
                print(
                    f"\n[Time Budget Guard] Next epoch (~{avg_ep_s / 60:.1f} min) would exceed "
                    f"{args.max_hours:.2f}h limit. Stopping cleanly."
                )
                break

        ep_t0 = time.time()
        train_loss = train_one_epoch(
            net=net,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=scheduler.T_max,
            is_hero=is_hero,
            amp=not args.no_amp,
            grad_accum=args.grad_accum,
            log_interval=args.log_interval,
            check_vram=(epoch == start_epoch),
            max_vram_gb=args.max_vram_gb,
        )

        # Dynamically adapt CosineAnnealing T_max after Epoch 1 so LR finishes smoothly within budget
        ep_train_eval_start = time.time()
        val_m = evaluate(net, val_loader, criterion, device, is_hero=is_hero, amp=not args.no_amp)
        ep_s = time.time() - ep_t0
        epoch_times.append(ep_s)

        if (not tmax_adapted) and args.auto_tmax and epoch == 1:
            projected = max(2, min(args.epochs, int((max_budget_s * 0.95) / max(1.0, ep_s))))
            if projected < args.epochs:
                scheduler.T_max = projected
                print(
                    f"  [Auto-Schedule] Epoch 1 took {ep_s:.1f}s ({ep_s / 60:.2f} min). "
                    f"Adapting CosineAnnealing T_max -> {projected} epochs to complete within {args.max_hours:.2f}h."
                )
            tmax_adapted = True

        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]
        total_elapsed_s = prior_elapsed_s + (time.time() - session_start)
        rem_h = max(0.0, (max_budget_s - total_elapsed_s) / 3600.0)

        print(
            f"--> [Epoch {epoch:03d}/{scheduler.T_max:03d} | {ep_s:.1f}s | Elapsed: {total_elapsed_s/3600:.2f}h | Rem: {rem_h:.2f}h] "
            f"Train Loss: {train_loss:.5f} | Val PSNR: {val_m['psnr']:.2f} dB (MM: {val_m['psnr_mm']:.2f}) | "
            f"Val SSIM: {val_m['ssim']:.4f} | LR: {curr_lr:.2e}"
        )

        if val_m["psnr"] > best_val_psnr:
            best_val_psnr = val_m["psnr"]
            best_val_ssim = val_m["ssim"]
            torch.save({
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "val_psnr": best_val_psnr,
                "val_ssim": best_val_ssim,
                "val_psnr_mm": val_m["psnr_mm"],
                "args": vars(args),
            }, best_path)
            print(f"    [* BEST PAPER CHECKPOINT SAVED] Val PSNR: {best_val_psnr:.2f} dB -> {best_path}")

        torch.save({
            "epoch": epoch,
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_psnr": best_val_psnr,
            "best_val_ssim": best_val_ssim,
            "elapsed_s": total_elapsed_s,
            "tmax_adapted": tmax_adapted,
            "args": vars(args),
        }, latest_path)

        if epoch % args.save_every == 0:
            snap = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pth")
            torch.save({"epoch": epoch, "state_dict": net.state_dict()}, snap)

        if epoch >= scheduler.T_max:
            print(f"\n[Schedule Complete] Reached target epoch {scheduler.T_max}.")
            break

    # Final evaluation on all 429 unseen LSUI test images using best checkpoint
    print("\n" + "=" * 84)
    print(f" FINAL LSUI TEST EVALUATION (429 Images): {model_label.upper()}")
    print("=" * 84)
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        net.load_state_dict(best_ckpt["state_dict"])
        best_ep = best_ckpt["epoch"]
    else:
        best_ep = 0

    test_m = evaluate(net, test_loader, criterion, device, is_hero=is_hero, amp=not args.no_amp)
    total_hours = (prior_elapsed_s + (time.time() - session_start)) / 3600.0
    print(f" Best Epoch:         {best_ep}")
    print(f" Total Train Time:   {total_hours:.2f} hours")
    print(f" Total Test Images:  {test_m['samples_evaluated']}")
    print(f" Average PSNR:       {test_m['psnr']:.2f} dB")
    print(f" Mean-Matched PSNR:  {test_m['psnr_mm']:.2f} dB")
    print(f" Average SSIM:       {test_m['ssim']:.4f}")
    print(f" Average Latency:    {test_m['latency_ms']:.2f} ms/img")
    print(f" Full Log Saved:     {os.path.abspath(log_path)}")
    print("=" * 84)

    summary = {
        "model": args.model,
        "label": model_label,
        "paper_ref": paper_ref,
        "params_M": round(params_m, 3),
        "best_epoch": best_ep,
        "total_hours": round(total_hours, 3),
        "best_val_psnr": round(best_val_psnr, 4),
        "best_val_ssim": round(best_val_ssim, 4),
        "test_psnr": round(test_m["psnr"], 4),
        "test_psnr_mm": round(test_m["psnr_mm"], 4),
        "test_ssim": round(test_m["ssim"], 4),
        "test_latency_ms": round(test_m["latency_ms"], 2),
        "test_images": test_m["samples_evaluated"],
    }
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved test summary JSON -> {results_json}")


if __name__ == "__main__":
    main()
