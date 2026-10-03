"""
WaterMamba Testing and Evaluation Pipeline.
Paper: arXiv:2405.08419 - WaterMamba: Visual State Space Model for Underwater Image Enhancement

Uses shared core components from core.py (TeeLogger, metrics, datasets, image exports).
"""

import os
import time
import argparse
import torch

try:
    from .watermamba_model import WaterMamba, HAS_MAMBA
    from .core import setup_tee_logger, calc_psnr, calc_ssim, get_dataloader, build_model, save_comparison_triplet
except (ImportError, ValueError):
    from watermamba_model import WaterMamba, HAS_MAMBA
    from core import setup_tee_logger, calc_psnr, calc_ssim, get_dataloader, build_model, save_comparison_triplet


def test(args):
    device_name = "cpu" if args.cpu else "cuda"
    net, device = build_model(
        dim=args.dim,
        img_size=args.img_size,
        blocks=args.blocks,
        checkpoint_path=args.checkpoint,
        device=device_name,
        scan_mode=args.scan_mode
    )

    print("\n" + "=" * 80)
    print(" STARTING WATERMAMBA TESTING & EVALUATION")
    print(f" Device:         {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
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
        max_samples=args.max_samples
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
    print(" WATERMAMBA TEST RESULTS SUMMARY")
    print(f" Total Test Images:  {count}")
    print(f" Average PSNR:       {mean_psnr:.2f} dB")
    print(f" Average SSIM:       {mean_ssim:.4f}")
    print(f" Average Latency:    {avg_latency:.2f} ms per image")
    print(f" Visual Outputs:     {os.path.abspath(vis_dir)}")
    print("=" * 80 + "\n")

    return {"psnr": mean_psnr, "ssim": mean_ssim, "latency_ms": avg_latency}


def parse_arguments():
    parser = argparse.ArgumentParser(description="WaterMamba Testing Suite")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/watermamba/watermamba_best.pth", help="Checkpoint to test")
    parser.add_argument("--dim", type=int, default=32, help="Base feature dimension C (default: 32)")
    parser.add_argument("--img_size", type=int, default=256, help="Square input resolution (default: 256)")
    parser.add_argument("--blocks", type=int, default=1, help="SCOSS blocks per stage (default: 1)")
    parser.add_argument("--scan_mode", type=str, choices=["auto", "2d", "1d"], default="auto",
                        help="Scan mode: 2d (fast spatial coordinate sweeps), 1d (flattened 4-way), auto")

    parser.add_argument("--dataset_type", type=str, choices=["lsui", "folder"], default="lsui")
    parser.add_argument("--data_dir", type=str, default="dataset/LSUI", help="Dataset root directory")
    parser.add_argument("--splits_csv", type=str, default=None, help="Path to splits csv")
    parser.add_argument("--results_dir", type=str, default="results/watermamba", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for Tee run logs")
    parser.add_argument("--num_visuals", type=int, default=10, help="Number of comparison images to save")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap test samples for fast evaluation")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")

    return parser.parse_args()


def main():
    args = parse_arguments()
    log_path = setup_tee_logger(log_dir=args.log_dir, run_name="watermamba_test")
    print(f"Command line arguments: {vars(args)}")

    try:
        test(args)
    finally:
        print(f"\n[Done] Testing finished. Full logs saved at: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
