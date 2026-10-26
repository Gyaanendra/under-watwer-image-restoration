"""
Evaluation Metrics, Memory Guard, and Experiment TeeLogger.
Includes:
1. Full-reference PSNR and SSIM.
2. Mean-Matched PSNR (normalizes global luminance offset before scoring).
3. ExperimentTeeLogger for dual console and file logging.
4. Step-1 GPU VRAM guardrail (aborts if memory exceeds budget).
5. Experiment Matrix CSV/JSON logger.
"""

import os
import sys
import math
import json
import time
from datetime import datetime
import pandas as pd
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Quantitative Fidelity Metrics
# --------------------------------------------------------------------------- #
def calc_psnr(pred, target, eps=1e-8):
    """Computes Peak Signal-to-Noise Ratio (dB) on tensors clamped in [0, 1]."""
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    mse = F.mse_loss(pred, target).item()
    if mse < eps:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def calc_mean_matched_psnr(pred, target, eps=1e-6):
    """
    Computes Mean-Matched PSNR:
    Scales predicted image by target_mean / pred_mean before computing PSNR.
    Isolates structural and contrast fidelity from simple overall brightness shifts.
    """
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    m_pred = pred.mean(dim=(1, 2, 3), keepdim=True) + eps
    m_target = target.mean(dim=(1, 2, 3), keepdim=True)
    pred_matched = torch.clamp(pred * (m_target / m_pred), 0.0, 1.0)
    mse = F.mse_loss(pred_matched, target).item()
    if mse < 1e-8:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def _gaussian_kernel(size=11, sigma=1.5, channels=3, device="cpu"):
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    return w.expand(channels, 1, size, size).contiguous()


def calc_ssim(pred, target, win_size=11, sigma=1.5):
    """Computes Structural Similarity Index (SSIM)."""
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    b, c, h, w = pred.shape
    kernel = _gaussian_kernel(win_size, sigma, c, pred.device).to(pred.dtype)

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.conv2d(pred, kernel, padding=win_size // 2, groups=c)
    mu2 = F.conv2d(target, kernel, padding=win_size // 2, groups=c)

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sig1_sq = F.conv2d(pred * pred, kernel, padding=win_size // 2, groups=c) - mu1_sq
    sig2_sq = F.conv2d(target * target, kernel, padding=win_size // 2, groups=c) - mu2_sq
    sig12 = F.conv2d(pred * target, kernel, padding=win_size // 2, groups=c) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sig12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sig1_sq + sig2_sq + c2))
    return ssim_map.mean().item()


# --------------------------------------------------------------------------- #
# 2. Dual Console & File TeeLogger
# --------------------------------------------------------------------------- #
class ExperimentTeeLogger:
    """
    Duplicates stdout and stderr output to both terminal and a disk log file.
    Safely handles Windows UTF-8 encodings and flushes on every write.
    """
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self.log_file = open(log_path, "a", encoding="utf-8")
        self.log_path = log_path
        sys.stdout = self
        sys.stderr = self

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
        sys.stdout = self.terminal
        sys.stderr = self.terminal
        self.log_file.close()


# --------------------------------------------------------------------------- #
# 3. GPU VRAM Guardrail
# --------------------------------------------------------------------------- #
def check_vram_guard(step, max_vram_gb=7.0, abort_on_exceed=True):
    """
    Queries torch.cuda.max_memory_allocated() after step 1.
    If allocated memory exceeds max_vram_gb, raises a clear error to abort before crash.
    """
    if not torch.cuda.is_available():
        return 0.0

    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_gb = peak_mb / 1024.0

    if step == 1:
        print(f"[VRAM Check @ Step 1] Peak Memory Allocated: {peak_mb:.1f} MB ({peak_gb:.2f} GB)")

    if peak_gb > max_vram_gb and abort_on_exceed:
        msg = f"CRITICAL: VRAM budget exceeded ({peak_gb:.2f} GB > {max_vram_gb:.1f} GB threshold). Aborting run to avoid OOM crash."
        print(msg)
        raise RuntimeError(msg)

    return peak_mb


# --------------------------------------------------------------------------- #
# 4. Summary Matrix CSV/JSON Updater
# --------------------------------------------------------------------------- #
def record_experiment_result(matrix_csv_path, result_dict):
    """
    Appends or updates a row in the central experiment tracking CSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(matrix_csv_path)), exist_ok=True)
    cols = [
        "run_id", "group", "model", "keep_ccoss", "attn", "act", "norm", "scales", "scan",
        "res", "epochs", "batch_size", "params_M", "peak_vram_mb",
        "best_val_psnr", "best_val_ssim", "best_val_psnr_mm", "final_train_loss",
        "avg_sec_per_epoch", "status", "timestamp"
    ]

    result_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(matrix_csv_path):
        df = pd.read_csv(matrix_csv_path)
        # Update existing run_id if re-run, else append
        if result_dict["run_id"] in df["run_id"].values:
            df.loc[df["run_id"] == result_dict["run_id"]] = pd.DataFrame([result_dict])[cols].values
        else:
            df = pd.concat([df, pd.DataFrame([result_dict])[cols]], ignore_index=True)
    else:
        df = pd.DataFrame([result_dict])[cols]

    df.to_csv(matrix_csv_path, index=False)
    print(f"[Matrix Updated] Result recorded in: {matrix_csv_path}")
