"""
Core utilities shared between WaterMamba training and testing suites.
Includes:
- TeeLogger for dual terminal and file logging.
- Quantitative evaluation metrics (PSNR, SSIM).
- Dataset loaders for LSUI and generic paired image datasets.
- Model instantiation and weight loading utilities.
"""

import os
import sys
import glob
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.utils as vutils

try:
    from .watermamba_model import WaterMamba, HAS_MAMBA
except (ImportError, ValueError):
    from watermamba_model import WaterMamba, HAS_MAMBA


# =========================================================================== #
# 1. Tee Logging Infrastructure
# =========================================================================== #
class TeeLogger:
    """
    Duplicates stdout and stderr output to both the console terminal
    and a designated log file on disk. Handles Windows console encoding.
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
    Returns the absolute path to the newly created log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{run_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    tee = TeeLogger(log_path, mode="w")
    sys.stdout = tee
    sys.stderr = tee

    latest_log_path = os.path.join(log_dir, f"{run_name}_latest.log")
    try:
        with open(latest_log_path, "w", encoding="utf-8") as f:
            f.write(f"Latest run started at: {datetime.now().isoformat()}\n")
            f.write(f"Active log file: {log_path}\n")
    except Exception:
        pass

    print("=" * 80)
    print(f" WaterMamba Logger Initialized")
    print(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Log File:  {os.path.abspath(log_path)}")
    print("=" * 80)
    return log_path


# =========================================================================== #
# 2. Evaluation Metrics: PSNR and SSIM
# =========================================================================== #
def calc_psnr(pred, gt, max_val=1.0):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB.
    pred, gt: (B, C, H, W) in range [0, 1]
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
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


# =========================================================================== #
# 3. Dataset Classes & Loaders
# =========================================================================== #
class LSUIPairDataset(Dataset):
    """
    Dataset using LSUI layout with lsui_splits.csv.
    Reads from input/ and GT/ subfolders based on split name.
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

        inp_img = TF.resize(inp_img, [self.img_size, self.img_size])
        gt_img = TF.resize(gt_img, [self.img_size, self.img_size])

        inp_tensor = TF.to_tensor(inp_img)
        gt_tensor = TF.to_tensor(gt_img)

        # Paired augmentation for training split
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
    Fallback dataset for arbitrary paired directories (e.g., raw/ and ref/).
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


def get_dataloader(data_dir, split="train", dataset_type="lsui", splits_csv=None,
                   img_size=256, batch_size=2, num_workers=0, max_samples=None):
    """
    Unified DataLoader constructor for training, validation, and testing.
    """
    if splits_csv is None:
        default_csv = os.path.join(data_dir, "lsui_splits.csv")
        if os.path.exists(default_csv):
            splits_csv = default_csv

    if dataset_type == "lsui" or (splits_csv and os.path.exists(splits_csv)):
        ds = LSUIPairDataset(
            data_dir=data_dir,
            split=split,
            splits_csv=splits_csv,
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

    is_training = (split == "train")
    bs = batch_size if is_training else 1
    shuffle = is_training
    drop_last = (is_training and len(ds) >= bs)

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available()
    )


# =========================================================================== #
# 4. Model Loading & Image Export Helpers
# =========================================================================== #
def build_model(dim=32, img_size=256, blocks=1, checkpoint_path=None, device="cuda", scan_mode="auto"):
    """
    Constructs WaterMamba and optionally loads checkpoint weights.
    """
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    net = WaterMamba(dim=dim, img_size=img_size, blocks=blocks, scan_mode=scan_mode).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        net.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    elif checkpoint_path:
        print(f"[WARNING] Checkpoint '{checkpoint_path}' not found! Using initialized weights.")

    return net, device


def save_comparison_triplet(raw_tensor, enhanced_tensor, ref_tensor, save_path):
    """
    Saves a side-by-side [Input | Enhanced | Ground Truth] comparison image.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    # Concatenate horizontally along width dimension
    triplet = torch.cat([raw_tensor.cpu(), enhanced_tensor.cpu(), ref_tensor.cpu()], dim=2)
    vutils.save_image(triplet, save_path)
