"""
Standalone Smoke Test for WaterMamba (Moved to scratch directory).
Used for quick verification of forward/backward passes, GPU execution, metrics, and Tee logging.
"""

import os
import sys
import torch

# Add parent directory to sys.path to import watermamba_model and train_and_test
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from watermamba_model import WaterMamba, HAS_MAMBA
from train_and_test import setup_tee_logger, run_training, run_testing, parse_arguments


def main():
    args = parse_arguments()
    log_path = setup_tee_logger(log_dir=args.log_dir, run_name="watermamba_smoke_test")

    print("\n" + "=" * 80)
    print(" WATERMAMBA SCRATCH SMOKE TEST")
    print(f" Device:         {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f" Fast SSM Mamba: {HAS_MAMBA}")
    print("=" * 80)

    # Force fast smoke test settings
    args.img_size = 32
    args.dim = 16
    args.epochs = 2
    args.batch_size = 2
    args.max_samples = 4
    args.max_val_batches = 2
    args.num_visuals = 2

    print(f"\n[1/2] Running mini-training ({args.img_size}x{args.img_size}, {args.epochs} epochs)...")
    ckpt_path = run_training(args)

    print("\n[2/2] Running test evaluation with generated checkpoint...")
    metrics = run_testing(args, checkpoint_path=ckpt_path)

    print(f"\n[DONE] Smoke test passed! Metrics: PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ssim']:.4f}")
    print(f"Full logs saved at: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
