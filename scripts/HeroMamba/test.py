"""
Hero-Mamba Testing and Evaluation Pipeline.
Paper: arXiv:2404.16266 - High-Efficiency and Robust Underwater Image Restoration
with State Space Models and Physical Priors.

Reuses core data loaders, metrics, and image triplet exports from scripts.waterMamba.core.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.HeroMamba.heromamba_model import HeroMamba, HAS_MAMBA
from scripts.waterMamba.core import (
    setup_tee_logger,
    calc_psnr,
    calc_ssim,
    get_dataloader,
    save_comparison_triplet,
)


def test(args):
    device_name = "cpu" if args.cpu else "cuda"
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    dims = (args.base_dim, args.base_dim * 2, args.base_dim * 4, args.base_dim * 8)
    net = HeroMamba(dims=dims, scan_mode=args.scan_mode).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        net.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    elif args.checkpoint:
        print(f"[WARNING] Checkpoint '{args.checkpoint}' not found! Using initialized weights.")

    print("\n" + "=" * 80)
    print(" STARTING HERO-MAMBA TESTING & EVALUATION")
    print(f" Device:         {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f" Fast SSM Mamba: {HAS_MAMBA}")
    print(f" Scan Mode:      {args.scan_mode} (using 2D scan: {net.ss2d_rgb.use_2d})")
    print(f" Resolution:     {args.img_size}x{args.img_size}")
    print(f" Results Folder: {os.path.abspath(args.results_dir)}")
    print("=" * 80)

    net.eval()
    test_loader = get_dataloader(
        data_dir=args.data_dir,
        split="test",
        dataset_type=args.dataset_type,
        splits_csv=args.splits_csv,
        img_size=args.img_size,
        batch_size=1,
        num_workers=0,
        max_samples=args.max_samples,
    )

    os.makedirs(args.results_dir, exist_ok=True)
    vis_dir = os.path.join(args.results_dir, "visual_comparisons")
    os.makedirs(vis_dir, exist_ok=True)

    total_psnr = 0.0
    total_ssim = 0.0
    total_time = 0.0
    count = 0

    print("\n--- Running Inference on Test Dataset ---")
    with torch.no_grad():
        for i, (raw, ref, fnames) in enumerate(test_loader):
            raw = raw.to(device)
            ref = ref.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            enhanced = net(raw).clamp(0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_time = time.time() - t0

            psnr_val = calc_psnr(enhanced, ref)
            ssim_val = calc_ssim(enhanced, ref)

            total_psnr += psnr_val
            total_ssim += ssim_val
            total_time += infer_time
            count += 1

            if count <= args.num_visuals:
                fname = fnames[0] if isinstance(fnames, (list, tuple)) else f"sample_{count}.png"
                clean_fname = os.path.splitext(fname)[0]
                save_out = os.path.join(vis_dir, f"{clean_fname}_comparison.png")
                save_comparison_triplet(raw[0], enhanced[0], ref[0], save_out)

            if count % 10 == 0 or count == len(test_loader):
                print(
                    f"Sample {count:04d}/{len(test_loader):04d} | "
                    f"PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | "
                    f"Latency: {infer_time * 1000:.1f} ms"
                )

    mean_psnr = total_psnr / max(1, count)
    mean_ssim = total_ssim / max(1, count)
    avg_latency = (total_time / max(1, count)) * 1000

    print("\n" + "=" * 80)
    print(" HERO-MAMBA TEST RESULTS SUMMARY")
    print(f" Total Test Images:  {count}")
    print(f" Average PSNR:       {mean_psnr:.2f} dB")
    print(f" Average SSIM:       {mean_ssim:.4f}")
    print(f" Average Latency:    {avg_latency:.2f} ms per image")
    print(f" Visual Outputs:     {os.path.abspath(vis_dir)}")
    print("=" * 80 + "\n")

    return {"psnr": mean_psnr, "ssim": mean_ssim, "latency_ms": avg_latency}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hero-Mamba Testing Suite")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/heromamba/heromamba_best.pth", help="Checkpoint to test")
    parser.add_argument("--base_dim", type=int, default=32, help="Base feature dimension C (default: 32)")
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
    parser.add_argument("--results_dir", type=str, default="results/heromamba", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for Tee run logs")
    parser.add_argument("--num_visuals", type=int, default=10, help="Number of comparison images to save")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap test samples for fast evaluation")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")

    return parser.parse_args()


def main():
    args = parse_arguments()
    log_path = setup_tee_logger(log_dir=args.log_dir, run_name="heromamba_test")
    print(f"Command line arguments: {vars(args)}")

    try:
        test(args)
    finally:
        print(f"\n[Done] Testing finished. Full logs saved at: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
