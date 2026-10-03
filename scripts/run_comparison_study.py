"""
Runs the full comparative analysis between WaterMamba and Hero-Mamba:
1. Parses training and testing logs.
2. Plots training curves and test benchmarks.
3. Loads trained models and runs inference on full-resolution real test images.
4. Computes image-level PSNR, SSIM, and RGB channel intensity histograms.
5. Saves all figures to results/comparison_study/.
"""

import os
import sys
import re
import glob
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms.functional as TF

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.waterMamba.watermamba_model import WaterMamba
from scripts.HeroMamba.heromamba_model import HeroMamba
from scripts.waterMamba.core import calc_psnr, calc_ssim, build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

os.makedirs("results/comparison_study", exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. Parse Logs
# --------------------------------------------------------------------------- #
def parse_training_log(filepath, model_name):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    pattern = r'--> \[Epoch (\d+)/\d+ Completed in ([\d\.]+)s\] Train Loss: ([\d\.]+) \| Val Loss: ([\d\.]+) \| Val PSNR: ([\d\.]+) dB \| Val SSIM: ([\d\.]+)'
    matches = re.findall(pattern, text)
    records = []
    for m in matches:
        records.append({
            "model": model_name,
            "epoch": int(m[0]),
            "epoch_time_s": float(m[1]),
            "train_loss": float(m[2]),
            "val_loss": float(m[3]),
            "val_psnr": float(m[4]),
            "val_ssim": float(m[5])
        })
    return pd.DataFrame(records)

def parse_testing_log(filepath, model_name):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        t = f.read()
    psnr = re.search(r'Average PSNR:\s+([\d\.]+)\s+dB', t)
    ssim = re.search(r'Average SSIM:\s+([\d\.]+)', t)
    lat = re.search(r'Average Latency:\s+([\d\.]+)\s+ms', t)
    imgs = re.search(r'Total Test Images:\s+(\d+)', t)
    return {
        "model": model_name,
        "test_psnr": float(psnr.group(1)) if psnr else None,
        "test_ssim": float(ssim.group(1)) if ssim else None,
        "test_latency_ms": float(lat.group(1)) if lat else None,
        "test_images": int(imgs.group(1)) if imgs else None,
    }

df_wm = parse_training_log("logs/watermamba_train_20261003_021400.log", "WaterMamba")
df_hm = parse_training_log("logs/heromamba_train_20261003_102744.log", "Hero-Mamba")
df_train = pd.concat([df_wm, df_hm], ignore_index=True)

test_wm = parse_testing_log("logs/watermamba_test_20261003_100849.log", "WaterMamba")
test_hm = parse_testing_log("logs/heromamba_test_20261003_130130.log", "Hero-Mamba")

print("\n--- TEST BENCHMARK (429 Images) ---")
print(pd.DataFrame([test_wm, test_hm]).to_string(index=False))

# --------------------------------------------------------------------------- #
# 2. Plot Training & Test Curves
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("WaterMamba vs. Hero-Mamba: Comprehensive Training & Benchmark Analysis", fontsize=15, fontweight='bold', y=0.98)
colors = {"WaterMamba": "#1f77b4", "Hero-Mamba": "#ff7f0e"}

# Train loss
ax = axes[0, 0]
for name, group in df_train.groupby("model"):
    ax.plot(group["epoch"], group["train_loss"], marker='o', linewidth=2.5, label=f"{name} (Final: {group['train_loss'].iloc[-1]:.4f})", color=colors[name])
ax.set_title("Training Loss Convergence vs. Epochs", fontsize=13, fontweight='bold')
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Loss Value", fontsize=11)
ax.set_xticks(range(1, 11))
ax.legend(frameon=True)
ax.grid(True, linestyle='--', alpha=0.6)

# Val PSNR
ax = axes[0, 1]
for name, group in df_train.groupby("model"):
    ax.plot(group["epoch"], group["val_psnr"], marker='s', linewidth=2.5, label=f"{name} (Best: {group['val_psnr'].max():.2f} dB)", color=colors[name])
ax.axhline(26.89, color=colors["WaterMamba"], linestyle=':', alpha=0.7)
ax.axhline(23.30, color=colors["Hero-Mamba"], linestyle=':', alpha=0.7)
ax.set_title("Validation PSNR (dB) Progression", fontsize=13, fontweight='bold')
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("PSNR (dB)", fontsize=11)
ax.set_xticks(range(1, 11))
ax.legend(frameon=True)
ax.grid(True, linestyle='--', alpha=0.6)

# Val SSIM
ax = axes[1, 0]
for name, group in df_train.groupby("model"):
    ax.plot(group["epoch"], group["val_ssim"], marker='^', linewidth=2.5, label=f"{name} (Best: {group['val_ssim'].max():.4f})", color=colors[name])
ax.set_title("Validation SSIM Progression", fontsize=13, fontweight='bold')
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("SSIM Index", fontsize=11)
ax.set_xticks(range(1, 11))
ax.legend(frameon=True)
ax.grid(True, linestyle='--', alpha=0.6)

# Test Comparison
ax = axes[1, 1]
categories = ['Test PSNR (dB)', 'Test SSIM (x30)', 'Speed (img/s x10)']
wm_vals = [test_wm['test_psnr'], test_wm['test_ssim'] * 30, (1000.0 / test_wm['test_latency_ms']) * 10]
hm_vals = [test_hm['test_psnr'], test_hm['test_ssim'] * 30, (1000.0 / test_hm['test_latency_ms']) * 10]
x_pos = np.arange(len(categories))
width = 0.35

r1 = ax.bar(x_pos - width/2, wm_vals, width, label='WaterMamba', color=colors["WaterMamba"], alpha=0.9)
r2 = ax.bar(x_pos + width/2, hm_vals, width, label='Hero-Mamba', color=colors["Hero-Mamba"], alpha=0.9)
ax.set_title("Test Benchmark: Fidelity vs. Speed Trade-Off", fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(frameon=True)
ax.grid(True, linestyle='--', alpha=0.6)

for r in r1:
    h = r.get_height()
    ax.annotate(f'{h:.1f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for r in r2:
    h = r.get_height()
    ax.annotate(f'{h:.1f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("results/comparison_study/training_curves.png", dpi=300)
plt.close()
print("Saved: results/comparison_study/training_curves.png")

# --------------------------------------------------------------------------- #
# 3. Load Models
# --------------------------------------------------------------------------- #
net_wm, _ = build_model(dim=32, img_size=256, blocks=1, checkpoint_path="checkpoints/watermamba/watermamba_best.pth", device="cuda" if torch.cuda.is_available() else "cpu", scan_mode="auto")
net_wm.eval()

net_hm = HeroMamba(dims=(32, 64, 128, 256), scan_mode="auto").to(device)
if os.path.exists("checkpoints/heromamba/heromamba_best.pth"):
    ckpt_hm = torch.load("checkpoints/heromamba/heromamba_best.pth", map_location=device)
    net_hm.load_state_dict(ckpt_hm.get("state_dict", ckpt_hm))
net_hm.eval()

p_wm = sum(p.numel() for p in net_wm.parameters()) / 1e6
p_hm = sum(p.numel() for p in net_hm.parameters()) / 1e6
print(f"Parameters: WaterMamba = {p_wm:.2f} M | Hero-Mamba = {p_hm:.2f} M")

# --------------------------------------------------------------------------- #
# 4. Inference on Full-Resolution Real Images
# --------------------------------------------------------------------------- #
def enhance_image(net, pil_img, device):
    orig_w, orig_h = pil_img.size
    img_low = TF.resize(pil_img, [256, 256])
    t_low = TF.to_tensor(img_low).unsqueeze(0).to(device)
    with torch.no_grad():
        t_enh_low = net(t_low).clamp(0.0, 1.0)
    ratio_low = (t_enh_low + 1e-4) / (t_low + 1e-4)
    ratio_high = F.interpolate(ratio_low, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    t_orig = TF.to_tensor(pil_img).unsqueeze(0).to(device)
    enhanced_high = (t_orig * ratio_high).clamp(0.0, 1.0)
    enhanced_np = (enhanced_high.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(enhanced_np)

test_samples = ["116.jpg", "4182.jpg", "1117.jpg"]
comparison_results = []

fig, axes = plt.subplots(len(test_samples), 4, figsize=(18, 4.5 * len(test_samples)))

for row_idx, fname in enumerate(test_samples):
    raw_path = os.path.join("dataset/LSUI/input", fname)
    gt_path = os.path.join("dataset/LSUI/GT", fname)
    if not (os.path.exists(raw_path) and os.path.exists(gt_path)):
        continue
    raw_img = Image.open(raw_path).convert("RGB")
    gt_img = Image.open(gt_path).convert("RGB")
    w, h = raw_img.size

    enh_wm = enhance_image(net_wm, raw_img, device)
    enh_hm = enhance_image(net_hm, raw_img, device)

    t_wm = TF.to_tensor(enh_wm).unsqueeze(0).to(device)
    t_hm = TF.to_tensor(enh_hm).unsqueeze(0).to(device)
    t_gt = TF.to_tensor(gt_img).unsqueeze(0).to(device)

    psnr_wm = calc_psnr(t_wm, t_gt)
    ssim_wm = calc_ssim(t_wm, t_gt)
    psnr_hm = calc_psnr(t_hm, t_gt)
    ssim_hm = calc_ssim(t_hm, t_gt)

    comparison_results.append({
        "sample": fname,
        "resolution": f"{w}x{h}",
        "raw_img": raw_img,
        "wm_img": enh_wm,
        "hm_img": enh_hm,
        "gt_img": gt_img,
        "wm_psnr": psnr_wm,
        "wm_ssim": ssim_wm,
        "hm_psnr": psnr_hm,
        "hm_ssim": ssim_hm,
    })

    axes[row_idx, 0].imshow(raw_img)
    axes[row_idx, 0].set_title(f"Raw Input: {fname} ({w}x{h})", fontsize=11, fontweight='bold')
    axes[row_idx, 0].axis('off')

    axes[row_idx, 1].imshow(enh_wm)
    axes[row_idx, 1].set_title(f"WaterMamba\\nPSNR: {psnr_wm:.2f} dB | SSIM: {ssim_wm:.4f}", fontsize=11, fontweight='bold', color='#1f77b4')
    axes[row_idx, 1].axis('off')

    axes[row_idx, 2].imshow(enh_hm)
    axes[row_idx, 2].set_title(f"Hero-Mamba\\nPSNR: {psnr_hm:.2f} dB | SSIM: {ssim_hm:.4f}", fontsize=11, fontweight='bold', color='#ff7f0e')
    axes[row_idx, 2].axis('off')

    axes[row_idx, 3].imshow(gt_img)
    axes[row_idx, 3].set_title(f"Ground Truth Reference ({w}x{h})", fontsize=11, fontweight='bold', color='green')
    axes[row_idx, 3].axis('off')

plt.tight_layout()
plt.savefig("results/comparison_study/real_image_comparisons.png", dpi=300)
plt.close()
print("Saved: results/comparison_study/real_image_comparisons.png")

# --------------------------------------------------------------------------- #
# 5. RGB Channel Histograms
# --------------------------------------------------------------------------- #
chosen = comparison_results[0]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
fig.suptitle(f"RGB Channel Intensity Histograms: {chosen['sample']} ({chosen['resolution']})", fontsize=15, fontweight='bold', y=1.02)

images_dict = {
    "1. Raw Underwater Input": chosen["raw_img"],
    f"2. WaterMamba (PSNR: {chosen['wm_psnr']:.1f} dB)": chosen["wm_img"],
    f"3. Hero-Mamba (PSNR: {chosen['hm_psnr']:.1f} dB)": chosen["hm_img"],
    "4. Ground Truth Target": chosen["gt_img"]
}

stats = []
for idx, (title, img) in enumerate(images_dict.items()):
    ax = axes[idx]
    arr = np.array(img)
    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()

    stats.append({
        "Image": title.split(":")[0],
        "Red Mean": f"{r.mean():.1f}",
        "Green Mean": f"{g.mean():.1f}",
        "Blue Mean": f"{b.mean():.1f}",
        "R/B Ratio": f"{r.mean() / (b.mean() + 1e-4):.2f}"
    })

    ax.hist(r, bins=64, range=(0, 255), color='red', alpha=0.5, label=f'Red (μ={r.mean():.0f})', density=True)
    ax.hist(g, bins=64, range=(0, 255), color='green', alpha=0.4, label=f'Green (μ={g.mean():.0f})', density=True)
    ax.hist(b, bins=64, range=(0, 255), color='blue', alpha=0.3, label=f'Blue (μ={b.mean():.0f})', density=True)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Pixel Intensity (0 - 255)", fontsize=10)
    if idx == 0:
        ax.set_ylabel("Normalized Density", fontsize=10)
    ax.legend(loc='upper right', fontsize=8.5)
    ax.set_xlim(0, 255)

plt.tight_layout()
plt.savefig("results/comparison_study/rgb_histograms.png", dpi=300)
plt.close()
print("Saved: results/comparison_study/rgb_histograms.png")

print("\n--- RGB CHANNEL STATS ---")
print(pd.DataFrame(stats).to_string(index=False))

print("\n[Done] All comparison assets generated successfully!")
