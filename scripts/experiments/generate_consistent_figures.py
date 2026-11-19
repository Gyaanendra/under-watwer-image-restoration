"""
Generate visually consistent publication-quality figures for IEEE conference paper:
- Uniform image dimensions (400x300) across all scenes and models.
- Clean row titles (Coral Reef, Turbid Lionfish, Pelagic Open-Water) instead of raw filenames.
- Consistent model names (1D Scan, Pooled 2D, W-1a SCA, H-1b RC Gate, H-D3-Ada, Ground Truth).
- Clean RGB histograms for the Coral Reef scene without raw filename in title.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Add scripts directory to path
PROJECT_ROOT = Path("E:/sem5_project")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/HeroMamba"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/waterMamba"))

from scripts.waterMamba.watermamba_model import WaterMamba
from scripts.waterMamba.core import calc_psnr, calc_ssim, build_model
from scripts.HeroMamba.heromamba_model import HeroMamba
from models_factory import build_experiment_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Models
models_dict = {}

# WaterMamba 1D Scan
wm_paper = WaterMamba(dim=32, img_size=256, blocks=1, scan_mode="1d")
ckpt = torch.load("checkpoints/paper_exact/watermamba_paper/best.pth", map_location=device, weights_only=False)
wm_paper.load_state_dict(ckpt["state_dict"])
wm_paper = wm_paper.to(device).eval()
models_dict["WaterMamba 1D"] = wm_paper

# WaterMamba Pooled 2D
wm_base, _ = build_model(dim=32, img_size=256, blocks=1, checkpoint_path="checkpoints/watermamba/watermamba_best.pth", device=device, scan_mode="auto")
wm_base = wm_base.to(device).eval()
models_dict["WaterMamba 2D"] = wm_base

# Hero-Mamba 1D Scan
hm_paper = HeroMamba(dims=(32, 64, 128, 256), scan_mode="1d")
ckpt = torch.load("checkpoints/paper_exact/heromamba_paper/best.pth", map_location=device, weights_only=False)
hm_paper.load_state_dict(ckpt["state_dict"])
hm_paper = hm_paper.to(device).eval()
models_dict["Hero-Mamba 1D"] = hm_paper

# Hero-Mamba Pooled 2D
hm_base = HeroMamba(dims=(32, 64, 128, 256), scan_mode="auto")
ckpt = torch.load("checkpoints/heromamba/heromamba_best.pth", map_location=device, weights_only=False)
hm_base.load_state_dict(ckpt.get("state_dict", ckpt))
hm_base = hm_base.to(device).eval()
models_dict["Hero-Mamba 2D"] = hm_base

# W-1a (SCA)
w_1a = build_experiment_model("wm", keep_ccoss=0, attn="sca", act="base")
ckpt = torch.load("checkpoints/experiments/W-1a/best_model.pth", map_location=device, weights_only=False)
w_1a.load_state_dict(ckpt.get("model_state_dict", ckpt))
w_1a = w_1a.to(device).eval()
models_dict["W-1a (SCA)"] = w_1a

# H-1b (RC Gate)
h_1b = build_experiment_model("hm", attn="rc", act="base", norm="bn", scales=0)
ckpt = torch.load("checkpoints/experiments/H-1b/best_model.pth", map_location=device, weights_only=False)
h_1b.load_state_dict(ckpt.get("model_state_dict", ckpt))
h_1b = h_1b.to(device).eval()
models_dict["H-1b (RC Gate)"] = h_1b

# H-D3-Ada (LN Decoder)
h_d3 = build_experiment_model("hm", attn="none", act="adaptivegate", norm="ln", scales=1)
ckpt = torch.load("checkpoints/experiments/H-D3-Ada/best_model.pth", map_location=device, weights_only=False)
h_d3.load_state_dict(ckpt.get("model_state_dict", ckpt))
h_d3 = h_d3.to(device).eval()
models_dict["H-D3-Ada"] = h_d3

print(f"Loaded {len(models_dict)} models successfully.")

# Inference Function with Bilateral Ratio Map
def enhance_image(net, pil_img, dev):
    orig_w, orig_h = pil_img.size
    img_low = TF.resize(pil_img, [256, 256])
    t_low = TF.to_tensor(img_low).unsqueeze(0).to(dev)
    with torch.no_grad():
        t_enh_low = net(t_low).clamp(0.0, 1.0)
    ratio_low = (t_enh_low + 1e-4) / (t_low + 1e-4)
    ratio_high = F.interpolate(ratio_low, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    t_orig = TF.to_tensor(pil_img).unsqueeze(0).to(dev)
    enhanced_high = (t_orig * ratio_high).clamp(0.0, 1.0)
    enhanced_np = (enhanced_high.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(enhanced_np)

def evaluate_pair(raw_pil, gt_pil, net, dev):
    enh_pil = enhance_image(net, raw_pil, dev)
    t_enh = TF.to_tensor(enh_pil).unsqueeze(0).to(dev)
    t_gt = TF.to_tensor(gt_pil).unsqueeze(0).to(dev)
    psnr = calc_psnr(t_enh, t_gt)
    ssim = calc_ssim(t_enh, t_gt)
    return enh_pil, psnr, ssim

# Samples configuration
# Target uniform size: 400x300 (4:3 aspect ratio)
TARGET_SIZE = (400, 300)

scenes = [
    {
        "name": "Coral Reef Scene",
        "file": "116.jpg",
        "crop_box": (160, 140, 260, 240)  # [left, top, right, bottom]
    },
    {
        "name": "Turbid Lionfish Scene",
        "file": "4182.jpg",
        "crop_box": (60, 70, 160, 170)
    },
    {
        "name": "Pelagic Open-Water Scene",
        "file": "1117.jpg",
        "crop_box": (120, 80, 220, 180)
    }
]

model_keys = list(models_dict.keys())
n_cols = len(model_keys) + 2  # Raw + 7 models + GT = 9 columns
n_rows = len(scenes)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(27, 2.35 * n_rows))

all_samples_res = []

for row_idx, sc in enumerate(scenes):
    raw_path = PROJECT_ROOT / "dataset/LSUI/input" / sc["file"]
    gt_path = PROJECT_ROOT / "dataset/LSUI/GT" / sc["file"]
    raw_img = Image.open(raw_path).convert("RGB")
    gt_img = Image.open(gt_path).convert("RGB")
    
    # Standardize to uniform TARGET_SIZE (400, 300)
    raw_img = raw_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    gt_img = gt_img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    
    sample_res = {"name": sc["name"], "raw_img": raw_img, "gt_img": gt_img, "enh_dict": {}, "metrics": {}}
    
    # Column 0: Raw Input
    axes[row_idx, 0].imshow(raw_img)
    axes[row_idx, 0].set_title(f"Raw Input\n{sc['name']}", fontsize=10.5, fontweight="bold", pad=4)
    axes[row_idx, 0].axis("off")
    
    # Columns 1 to 7: Models
    for m_idx, m_name in enumerate(model_keys):
        col = m_idx + 1
        net = models_dict[m_name]
        enh_img, p, s = evaluate_pair(raw_img, gt_img, net, device)
        sample_res["enh_dict"][m_name] = enh_img
        sample_res["metrics"][m_name] = (p, s)
        
        axes[row_idx, col].imshow(enh_img)
        # Clean title color: red for top 1D scan, blue for pooled 2D, dark gray for ablations
        if "1D" in m_name:
            c = "#c81e1e"
        elif "2D" in m_name:
            c = "#1e3cc8"
        elif "W-1a" in m_name:
            c = "#990099"
        else:
            c = "#222222"
            
        axes[row_idx, col].set_title(f"{m_name}\n{p:.2f} dB | {s:.3f}", fontsize=10, fontweight="bold", color=c, pad=4)
        axes[row_idx, col].axis("off")
        
    # Column 8: Ground Truth
    axes[row_idx, n_cols - 1].imshow(gt_img)
    axes[row_idx, n_cols - 1].set_title(f"Ground Truth\nReference", fontsize=10.5, fontweight="bold", color="#008000", pad=4)
    axes[row_idx, n_cols - 1].axis("off")
    
    all_samples_res.append(sample_res)

plt.subplots_adjust(wspace=0.02, hspace=0.12, left=0.01, right=0.99, top=0.93, bottom=0.02)

out_fig1 = PROJECT_ROOT / "reports/figures/final_real_image_comparisons.png"
out_fig1_jpg = PROJECT_ROOT / "reports/figures/final_real_image_comparisons.jpg"
plt.savefig(out_fig1, dpi=300, bbox_inches="tight")
plt.savefig(out_fig1_jpg, dpi=300, bbox_inches="tight")
import shutil
shutil.copy(out_fig1, PROJECT_ROOT / "reports/figs/final_real_image_comparisons.png")
shutil.copy(out_fig1_jpg, PROJECT_ROOT / "reports/figs/final_real_image_comparisons.jpg")
plt.close()
print(f"Saved uniform real image comparisons to: {out_fig1}")

# 2. Generate Clean Zoom Crops
# Columns: Raw, WaterMamba 2D, WaterMamba 1D, Hero-Mamba 1D, W-1a (SCA), Ground Truth (6 columns)
zoom_cols = ["Raw", "WaterMamba 2D", "WaterMamba 1D", "Hero-Mamba 1D", "W-1a (SCA)", "Ground Truth"]
fig_z, axes_z = plt.subplots(n_rows, len(zoom_cols), figsize=(16, 2.8 * n_rows))

for row_idx, sc in enumerate(scenes):
    s_res = all_samples_res[row_idx]
    box = sc["crop_box"]
    
    crops = [
        s_res["raw_img"].crop(box),
        s_res["enh_dict"]["WaterMamba 2D"].crop(box),
        s_res["enh_dict"]["WaterMamba 1D"].crop(box),
        s_res["enh_dict"]["Hero-Mamba 1D"].crop(box),
        s_res["enh_dict"]["W-1a (SCA)"].crop(box),
        s_res["gt_img"].crop(box)
    ]
    
    for c_idx, (col_name, crp) in enumerate(zip(zoom_cols, crops)):
        ax = axes_z[row_idx, c_idx]
        ax.imshow(crp)
        if row_idx == 0:
            ax.set_title(col_name, fontsize=11, fontweight="bold", pad=4)
        if c_idx == 0:
            ax.set_ylabel(sc["name"].replace(" Scene", ""), fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

plt.subplots_adjust(wspace=0.04, hspace=0.10, left=0.06, right=0.99, top=0.93, bottom=0.02)
out_zoom = PROJECT_ROOT / "reports/figures/zoom_crops.jpg"
out_zoom_png = PROJECT_ROOT / "reports/figures/zoom_crops.png"
plt.savefig(out_zoom, dpi=300, bbox_inches="tight")
plt.savefig(out_zoom_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved clean zoom crops to: {out_zoom}")

# 3. Generate Clean RGB Histograms (Coral Reef Scene)
print("Generating RGB Histograms...")
coral_res = all_samples_res[0]
hist_dict = {
    "Raw Input": coral_res["raw_img"],
    f"WaterMamba 1D ({coral_res['metrics']['WaterMamba 1D'][0]:.1f} dB)": coral_res["enh_dict"]["WaterMamba 1D"],
    f"WaterMamba 2D ({coral_res['metrics']['WaterMamba 2D'][0]:.1f} dB)": coral_res["enh_dict"]["WaterMamba 2D"],
    f"Hero-Mamba 1D ({coral_res['metrics']['Hero-Mamba 1D'][0]:.1f} dB)": coral_res["enh_dict"]["Hero-Mamba 1D"],
    f"Hero-Mamba 2D ({coral_res['metrics']['Hero-Mamba 2D'][0]:.1f} dB)": coral_res["enh_dict"]["Hero-Mamba 2D"],
    f"W-1a SCA ({coral_res['metrics']['W-1a (SCA)'][0]:.1f} dB)": coral_res["enh_dict"]["W-1a (SCA)"],
    f"H-1b RC Gate ({coral_res['metrics']['H-1b (RC Gate)'][0]:.1f} dB)": coral_res["enh_dict"]["H-1b (RC Gate)"],
    "Ground Truth": coral_res["gt_img"]
}

fig_h, axes_h = plt.subplots(1, len(hist_dict), figsize=(24, 3.4), sharey=True)
fig_h.suptitle("Spectral RGB Channel Distributions: Coral Reef Scene", fontsize=13, fontweight="bold", y=1.02)

for idx, (title, img) in enumerate(hist_dict.items()):
    ax = axes_h[idx]
    arr = np.array(img)
    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()
    
    ax.hist(r, bins=64, range=(0, 255), color="red", alpha=0.5, label=f"R (μ={r.mean():.0f})", density=True)
    ax.hist(g, bins=64, range=(0, 255), color="green", alpha=0.4, label=f"G (μ={g.mean():.0f})", density=True)
    ax.hist(b, bins=64, range=(0, 255), color="blue", alpha=0.3, label=f"B (μ={b.mean():.0f})", density=True)
    
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel Value", fontsize=9)
    if idx == 0:
        ax.set_ylabel("Normalized Density", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
out_hist = PROJECT_ROOT / "reports/figures/final_rgb_histograms.png"
out_hist_jpg = PROJECT_ROOT / "reports/figures/final_rgb_histograms.jpg"
plt.savefig(out_hist, dpi=300, bbox_inches="tight")
plt.savefig(out_hist_jpg, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved clean RGB histograms to: {out_hist}")
print("All consistent figures successfully created!")
