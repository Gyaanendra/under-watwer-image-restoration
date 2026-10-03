"""
WaterMamba Training and Testing Pipeline with Tee Logging.
Paper: arXiv:2405.08419 - WaterMamba: Visual State Space Model for Underwater Image Enhancement

Features:
- Dual console and file logging via TeeLogger.
- Support for LSUI split CSV (train/val/test) and custom paired image folders.
- Full training loop with AdamW, Cosine Annealing, and checkpointing.
- Model validation and test evaluation with quantitative PSNR and SSIM.
- Visual side-by-side artifact export (Input | Ground Truth | Enhanced).
- CLI interface for train, test, train_test, and smoke_test modes.
"""

import os
import sys
import glob
import time
import math
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.utils as vutils

# Support importing model from either relative or absolute directory
try:
    from .watermamba_model import WaterMamba, HAS_MAMBA
except (ImportError, ValueError):
    from watermamba_model import WaterMamba, HAS_MAMBA


# =========================================================================== #
# 1. Tee Logger: Logs simultaneously to terminal and log files
# =========================================================================== #
class TeeLogger:
    """
    Duplicates stdout and stderr output to both the console terminal
    and a designated log file on disk.
    Handles Windows CP1252 console encoding gracefully.
    """
    def __init__(self, log_path, mode="a"):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self.log_file = open(log_path, mode, encoding="utf-8")
        self.log_path = log_path

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal, "encoding", "utf-8") or "ascii"
            safe_msg = message.encode(encoding, errors="replace").decode(encoding)
            self.terminal.write(safe_msg)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.flush()
            self.log_file.close()


def setup_tee_logger(log_dir="logs", run_name="watermamba_run"):
    """
    Initializes TeeLogger and redirects sys.stdout and sys.stderr.
    Returns the absolute path of the generated log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{run_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    tee = TeeLogger(log_path, mode="w")
    sys.stdout = tee
    sys.stderr = tee

    # Also maintain a 'watermamba_latest.log' pointer/file for ease of inspection
    latest_log_path = os.path.join(log_dir, "watermamba_latest.log")
    try:
        with open(latest_log_path, "w", encoding="utf-8") as f:
            f.write(f"Latest run started at: {datetime.now().isoformat()}\n")
            f.write(f"Active log file: {log_path}\n")
    except Exception:
        pass

    print("=" * 80)
    print(f" WaterMamba Logger initialized")
    print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Log File:  {os.path.abspath(log_path)}")
    print("=" * 80)
    return log_path


# =========================================================================== #
# 2. Metrics: PSNR and SSIM (Pure PyTorch implementation)
# =========================================================================== #
def calc_psnr(pred, gt, max_val=1.0):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB.
    pred, gt: (B, C, H, W) in [0, 1]
    """
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(max_val / math.sqrt(mse.item()))


def _gaussian_window(window_size=11, sigma=1.5, channel=3):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    g = g / g.sum()
    w_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return w_2d.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)


def calc_ssim(pred, gt, window_size=11, sigma=1.5):
    """
    Computes Structural Similarity Index (SSIM).
    pred, gt: (B, C, H, W) in range [0, 1]
    """
    channel = pred.size(1)
    w = _gaussian_window(window_size, sigma, channel).to(pred.device, dtype=pred.dtype)
    mu1 = F.conv2d(pred, w, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(gt, w, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, w, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, w, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * gt, w, padding=window_size // 2, groups=channel) - mu1_mu2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


# =========================================================================== #
# 3. Dataset Classes: LSUI Split CSV and Generic Folder Pair
# =========================================================================== #
class LSUIPairDataset(Dataset):
    """
    Dataset using LSUI layout with lsui_splits.csv.
    Expected folder structure:
        data_dir/
            lsui_splits.csv
            input/<filename>
            GT/<filename>
    """
    def __init__(self, data_dir, split="train", splits_csv=None, img_size=256, max_samples=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size

        csv_file = Path(splits_csv) if splits_csv else self.data_dir / "lsui_splits.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Splits file not found: {csv_file}")

        import pandas as pd
        df = pd.read_csv(csv_file)
        split_df = df[df["split"].str.lower() == split.lower()]
        if len(split_df) == 0:
            raise ValueError(f"No samples found for split '{split}' in {csv_file}")

        self.samples = []
        for _, row in split_df.iterrows():
            fname = str(row["filename"])
            inp_p = self.data_dir / "input" / fname
            gt_p = self.data_dir / "GT" / fname
            if inp_p.exists() and gt_p.exists():
                self.samples.append((str(inp_p), str(gt_p), fname))

        if max_samples and max_samples > 0:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} pairs for split '{split}' from {self.data_dir.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp_path, gt_path, fname = self.samples[idx]
        inp_img = Image.open(inp_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        # Resize to fixed resolution required by model
        inp_img = TF.resize(inp_img, [self.img_size, self.img_size])
        gt_img = TF.resize(gt_img, [self.img_size, self.img_size])

        inp_tensor = TF.to_tensor(inp_img)
        gt_tensor = TF.to_tensor(gt_img)

        # Paired augmentation during training
        if self.split == "train":
            if torch.rand(1).item() < 0.5:
                inp_tensor = inp_tensor.flip(-1)
                gt_tensor = gt_tensor.flip(-1)
            if torch.rand(1).item() < 0.5:
                inp_tensor = inp_tensor.flip(-2)
                gt_tensor = gt_tensor.flip(-2)

        return inp_tensor, gt_tensor, fname


class GenericFolderPairDataset(Dataset):
    """
    Fallback dataset for custom paired directories (e.g., raw/ and ref/).
    """
    def __init__(self, root_dir, raw_sub="raw", ref_sub="ref", img_size=256, is_train=True, max_samples=None):
        super().__init__()
        self.root = Path(root_dir)
        self.img_size = img_size
        self.is_train = is_train

        raw_dir = self.root / raw_sub
        ref_dir = self.root / ref_sub

        raw_files = sorted(glob.glob(str(raw_dir / "*.*")))
        self.samples = []
        for r in raw_files:
            fname = Path(r).name
            target = ref_dir / fname
            if target.exists():
                self.samples.append((r, str(target), fname))

        if max_samples and max_samples > 0:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} pairs from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_p, ref_p, fname = self.samples[idx]
        raw_im = Image.open(raw_p).convert("RGB")
        ref_im = Image.open(ref_p).convert("RGB")

        raw_im = TF.resize(raw_im, [self.img_size, self.img_size])
        ref_im = TF.resize(ref_im, [self.img_size, self.img_size])

        raw_t = TF.to_tensor(raw_im)
        ref_t = TF.to_tensor(ref_im)

        if self.is_train and torch.rand(1).item() < 0.5:
            raw_t = raw_t.flip(-1)
            ref_t = ref_t.flip(-1)

        return raw_t, ref_t, fname


def get_dataloader(args, split="train"):
    """Constructs Dataset and DataLoader based on CLI args."""
    dataset_type = getattr(args, "dataset_type", "lsui")
    data_dir = args.data_dir
    img_size = args.img_size
    batch_size = args.batch_size if split == "train" else 1
    num_workers = getattr(args, "num_workers", 0)
    max_samples = getattr(args, f"max_{split}_samples", None)
    if max_samples is None and getattr(args, "max_samples", None):
        max_samples = args.max_samples

    if dataset_type == "lsui" or os.path.exists(os.path.join(data_dir, "lsui_splits.csv")):
        ds = LSUIPairDataset(
            data_dir=data_dir,
            split=split,
            splits_csv=getattr(args, "splits_csv", None),
            img_size=img_size,
            max_samples=max_samples
        )
    else:
        is_train = (split == "train")
        raw_sub = "input" if os.path.exists(os.path.join(data_dir, "input")) else "raw"
        ref_sub = "GT" if os.path.exists(os.path.join(data_dir, "GT")) else "ref"
        ds = GenericFolderPairDataset(
            root_dir=data_dir,
            raw_sub=raw_sub,
            ref_sub=ref_sub,
            img_size=img_size,
            is_train=is_train,
            max_samples=max_samples
        )

    shuffle = (split == "train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=(split == "train" and len(ds) >= batch_size),
        pin_memory=(torch.cuda.is_available())
    )


# =========================================================================== #
# 4. Training Engine
# =========================================================================== #
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
        loss = criterion(enhanced, ref)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            avg_loss = running_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(
                f"[Epoch {epoch:03d}/{total_epochs:03d}] "
                f"Batch {batch_idx + 1:04d}/{num_batches:04d} | "
                f"Loss: {loss.item():.5f} (Avg: {avg_loss:.5f}) | "
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

            loss = criterion(enhanced, ref)
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


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("\n" + "=" * 80)
    print(" STARTING WATERMAMBA TRAINING")
    print(f" Device:         {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f" Fast SSM Mamba: {HAS_MAMBA}")
    print(f" Resolution:     {args.img_size}x{args.img_size}")
    print(f" Base Dim:       {args.dim}")
    print(f" Batch Size:     {args.batch_size}")
    print(f" Learning Rate:  {args.lr}")
    print(f" Epochs:         {args.epochs}")
    print("=" * 80)

    # Initialize model
    net = WaterMamba(dim=args.dim, img_size=args.img_size, blocks=args.blocks).to(device)
    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f" Model initialized. Total parameters: {total_params:.3f} M")

    # Dataloaders
    train_loader = get_dataloader(args, split="train")
    val_loader = get_dataloader(args, split="val")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.L1Loss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_psnr = -1.0
    best_ckpt_path = os.path.join(args.checkpoint_dir, "watermamba_best.pth")
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "watermamba_latest.pth")

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
            log_interval=args.log_interval
        )
        scheduler.step()
        curr_lr = optimizer.param_groups[0]["lr"]

        # Validation step
        val_metrics = evaluate(net, val_loader, criterion, device, max_eval_batches=args.max_val_batches)
        epoch_time = time.time() - epoch_start

        print(
            f"--> [Epoch {epoch:03d}/{args.epochs:03d} Completed in {epoch_time:.1f}s] "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_metrics['loss']:.5f} | "
            f"Val PSNR: {val_metrics['psnr']:.2f} dB | "
            f"Val SSIM: {val_metrics['ssim']:.4f} | "
            f"LR: {curr_lr:.6e}"
        )

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "args": vars(args)
        }, latest_ckpt_path)

        # Check for best checkpoint
        if val_metrics["psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["psnr"]
            torch.save({
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "val_psnr": best_val_psnr,
                "val_ssim": val_metrics["ssim"],
                "args": vars(args)
            }, best_ckpt_path)
            print(f"    [* BEST MODEL SAVED] New best PSNR: {best_val_psnr:.2f} dB -> {best_ckpt_path}")

        # Periodic checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            p_path = os.path.join(args.checkpoint_dir, f"watermamba_epoch_{epoch:03d}.pth")
            torch.save({"epoch": epoch, "state_dict": net.state_dict()}, p_path)
            print(f"    [Checkpoint Saved] -> {p_path}")

    print("\n Training Complete!")
    print(f" Best Validation PSNR: {best_val_psnr:.2f} dB")
    return best_ckpt_path


# =========================================================================== #
# 5. Testing and Quantitative Evaluation Engine
# =========================================================================== #
def run_testing(args, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("\n" + "=" * 80)
    print(" STARTING WATERMAMBA TESTING & EVALUATION")
    print(f" Device:         {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f" Results Folder: {os.path.abspath(args.results_dir)}")
    print("=" * 80)

    # Initialize model
    net = WaterMamba(dim=args.dim, img_size=args.img_size, blocks=args.blocks).to(device)

    # Load checkpoint
    ckpt_to_load = checkpoint_path or args.checkpoint
    if ckpt_to_load and os.path.exists(ckpt_to_load):
        print(f" Loading weights from: {ckpt_to_load}")
        ckpt = torch.load(ckpt_to_load, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        net.load_state_dict(state_dict)
        print(" Weights loaded successfully.")
    else:
        print(f" [WARNING] No checkpoint found at '{ckpt_to_load}'. Running test with current/initialized weights.")

    net.eval()
    test_loader = get_dataloader(args, split="test")

    os.makedirs(args.results_dir, exist_ok=True)
    vis_dir = os.path.join(args.results_dir, "visual_comparisons")
    os.makedirs(vis_dir, exist_ok=True)

    total_psnr = 0.0
    total_ssim = 0.0
    total_time = 0.0
    count = 0

    print("\n--- Running Inference & Evaluation on Test Set ---")
    with torch.no_grad():
        for i, (raw, ref, fnames) in enumerate(test_loader):
            raw = raw.to(device)
            ref = ref.to(device)

            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.time()
            enhanced = net(raw).clamp(0.0, 1.0)
            torch.cuda.synchronize() if device.type == "cuda" else None
            infer_time = time.time() - t0

            psnr_val = calc_psnr(enhanced, ref)
            ssim_val = calc_ssim(enhanced, ref)

            total_psnr += psnr_val
            total_ssim += ssim_val
            total_time += infer_time
            count += 1

            # Save comparison visualization for the first N samples
            if count <= args.num_visuals:
                fname = fnames[0] if isinstance(fnames, (list, tuple)) else f"sample_{count}.png"
                clean_fname = os.path.splitext(fname)[0]
                # Concat horizontally: [Input | Enhanced | Ground Truth]
                triplet = torch.cat([raw[0].cpu(), enhanced[0].cpu(), ref[0].cpu()], dim=2)
                save_out = os.path.join(vis_dir, f"{clean_fname}_comparison.png")
                vutils.save_image(triplet, save_out)

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


# =========================================================================== #
# 6. Main Entry Point and Argument Parser
# =========================================================================== #
def parse_arguments():
    parser = argparse.ArgumentParser(description="WaterMamba Training and Testing Suite")

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "train_test"],
        default="train_test",
        help="Pipeline execution mode: 'train', 'test', or 'train_test' (default: train_test)"
    )

    # Model architecture
    parser.add_argument("--dim", type=int, default=32, help="Base feature dimension C (default: 32)")
    parser.add_argument("--img_size", type=int, default=256, help="Square input image resolution (default: 256)")
    parser.add_argument("--blocks", type=int, default=1, help="SCOSS blocks per stage (default: 1)")

    # Data paths
    parser.add_argument("--dataset_type", type=str, choices=["lsui", "folder"], default="lsui")
    parser.add_argument("--data_dir", type=str, default="dataset/LSUI", help="Dataset root directory")
    parser.add_argument("--splits_csv", type=str, default="dataset/LSUI/lsui_splits.csv", help="Path to splits csv")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (paper uses 750)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default: 0)")

    # Output & Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/watermamba", help="Directory for model weights")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/watermamba/watermamba_best.pth", help="Checkpoint to test")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save run logs")
    parser.add_argument("--results_dir", type=str, default="results/watermamba", help="Directory to save test outputs")
    parser.add_argument("--save_every", type=int, default=5, help="Epoch interval to save checkpoint")
    parser.add_argument("--log_interval", type=int, default=10, help="Batch logging interval")
    parser.add_argument("--num_visuals", type=int, default=10, help="Number of test comparison images to save")

    # Debug / Quick-run controls
    parser.add_argument("--max_samples", type=int, default=None, help="Cap dataset samples for fast debugging")
    parser.add_argument("--max_val_batches", type=int, default=None, help="Cap validation batches")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # 1. Initialize Tee Logger into logs/ directory
    log_path = setup_tee_logger(log_dir=args.log_dir, run_name=f"watermamba_{args.mode}")

    print(f"Command line arguments: {vars(args)}")

    try:
        if args.mode == "train":
            run_training(args)
        elif args.mode == "test":
            run_testing(args)
        elif args.mode == "train_test":
            best_ckpt = run_training(args)
            run_testing(args, checkpoint_path=best_ckpt)
    finally:
        print(f"\n[Done] Run finished. Full logs saved at: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
