"""
Generate a unified, highly legible detail analysis figure for IEEE conference paper:
- Top 2 rows: Localized high-frequency zoom crops (Coral Reef & Turbid Lionfish)
  Columns: Raw Input, WaterMamba 1D, WaterMamba 2D, Hero-Mamba 2D, W-1a (SCA), Ground Truth
- Bottom row: Aligned Spectral RGB distributions for Coral Reef Scene across the exact same 6 columns
- Large bold typography, uniform patch dimensions, NO squishing/distortion.
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

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

# 1. Load the 4 evaluation models needed
print("Loading models...")
# WaterMamba 1D Scan (Published Baseline)
wm_1d = WaterMamba(dim=32, img_size=256, blocks=1, scan_mode="1d")
ckpt = torch.load("checkpoints/paper_exact/watermamba_paper/best.pth", map_location=device, weights_only=False)
wm_1d.load_state_dict(ckpt["state_dict"])
wm_1d = wm_1d.to(device).eval()

# WaterMamba Pooled 2D (Proposed Regime B)
wm_2d, _ = build_model(dim=32, img_size=256, blocks=1, checkpoint_path="checkpoints/watermamba/watermamba_best.pth", device=device, scan_mode="auto")
wm_2d = wm_2d.to(device).eval()

# Hero-Mamba Pooled 2D (Proposed Regime B)
hm_2d = HeroMamba(dims=(32, 64, 128, 256), scan_mode="auto")
ckpt = torch.load("checkpoints/heromamba/heromamba_best.pth", map_location=device, weights_only=False)
hm_2d.load_state_dict(ckpt.get("state_dict", ckpt))
hm_2d = hm_2d.to(device).eval()

# W-1a (Proposed SCA Ablation)
w_1a = build_experiment_model("wm", keep_ccoss=0, attn="sca", act="base")
ckpt = torch.load("checkpoints/experiments/W-1a/best_model.pth", map_location=device, weights_only=False)
w_1a.load_state_dict(ckpt.get("model_state_dict", ckpt))
w_1a = w_1a.to(device).eval()

models = {
    "WaterMamba (1D Scan)": wm_1d,
    "WaterMamba (Pooled 2D)": wm_2d,
    "Hero-Mamba (Pooled 2D)": hm_2d,
    "W-1a (SCA Attn)": w_1a
}

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

# Target scenes
TARGET_SIZE = (400, 300)
scenes = [
    {
        "row_label": "Coral Reef\nScene",
        "file": "116.jpg",
        "box": (165, 145, 265, 245) # 100x100 square
    },
    {
        "row_label": "Turbid\nLionfish",
        "file": "4182.jpg",
        "box": (60, 70, 160, 170)    # 100x100 square
    }
]

# Run enhancement for both scenes
scene_data = []
for sc in scenes:
    raw_p = PROJECT_ROOT / "dataset/LSUI/input" / sc["file"]
    gt_p = PROJECT_ROOT / "dataset/LSUI/GT" / sc["file"]
    raw_img = Image.open(raw_p).convert("RGB").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    gt_img = Image.open(gt_p).convert("RGB").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    
    enh_dict = {}
    for m_name, net in models.items():
        enh_dict[m_name] = enhance_image(net, raw_img, device)
        
    scene_data.append({
        "row_label": sc["row_label"],
        "box": sc["box"],
        "raw_img": raw_img,
        "gt_img": gt_img,
        "enh_dict": enh_dict
    })

print("Enhanced images generated. Creating unified detail figure...")

# Define columns (6 total)
col_headers = [
    "Raw Input",
    "WaterMamba (1D Scan)\n[Published Baseline]",
    "WaterMamba (Pooled 2D)\n[Proposed Regime B]",
    "Hero-Mamba (Pooled 2D)\n[Proposed Regime B]",
    "W-1a (SCA Attn)\n[Proposed Ablation]",
    "Ground Truth\n[Reference]"
]

col_colors = [
    "#222222",
    "#c81e1e", # red
    "#1e3cc8", # blue
    "#1e3cc8", # blue
    "#800080", # purple
    "#007700"  # green
]

# Create Figure: 3 rows x 6 columns
# Row 0: Coral Reef Crops
# Row 1: Turbid Lionfish Crops
# Row 2: Coral Reef RGB Histograms
fig = plt.figure(figsize=(19.0, 5.2))
gs = fig.add_gridspec(3, 6, height_ratios=[1.15, 1.15, 0.95], hspace=0.15, wspace=0.05, left=0.06, right=0.99, top=0.91, bottom=0.06)

# Rows 0 and 1: Crops
for row_idx, s_info in enumerate(scene_data):
    box = s_info["box"]
    images_in_row = [
        s_info["raw_img"].crop(box),
        s_info["enh_dict"]["WaterMamba (1D Scan)"].crop(box),
        s_info["enh_dict"]["WaterMamba (Pooled 2D)"].crop(box),
        s_info["enh_dict"]["Hero-Mamba (Pooled 2D)"].crop(box),
        s_info["enh_dict"]["W-1a (SCA Attn)"].crop(box),
        s_info["gt_img"].crop(box)
    ]
    
    for c_idx, img in enumerate(images_in_row):
        ax = fig.add_subplot(gs[row_idx, c_idx])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Column title only on row 0
        if row_idx == 0:
            ax.set_title(col_headers[c_idx], fontsize=11, fontweight="bold", color=col_colors[c_idx], pad=5)
            
        # Row label on column 0
        if c_idx == 0:
            ax.set_ylabel(s_info["row_label"], fontsize=12, fontweight="bold", rotation=0, labelpad=42, va="center")

# Row 2: Histograms for Coral Reef Scene
coral_s = scene_data[0]
coral_images = [
    coral_s["raw_img"],
    coral_s["enh_dict"]["WaterMamba (1D Scan)"],
    coral_s["enh_dict"]["WaterMamba (Pooled 2D)"],
    coral_s["enh_dict"]["Hero-Mamba (Pooled 2D)"],
    coral_s["enh_dict"]["W-1a (SCA Attn)"],
    coral_s["gt_img"]
]

for c_idx, img in enumerate(coral_images):
    ax = fig.add_subplot(gs[2, c_idx])
    arr = np.array(img)
    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()
    
    # Plot histograms
    ax.hist(r, bins=48, range=(0, 255), color="#e6194b", alpha=0.55, density=True, label=f"R:{r.mean():.0f}")
    ax.hist(g, bins=48, range=(0, 255), color="#3cb44b", alpha=0.45, density=True, label=f"G:{g.mean():.0f}")
    ax.hist(b, bins=48, range=(0, 255), color="#4363d8", alpha=0.35, density=True, label=f"B:{b.mean():.0f}")
    
    ax.set_xlim(0, 255)
    ax.set_xticks([0, 128, 255])
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=8.5, loc="upper right", handlelength=0.8, framealpha=0.6, borderpad=0.2)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    if c_idx == 0:
        ax.set_ylabel("Spectral\nRGB Mean", fontsize=12, fontweight="bold", rotation=0, labelpad=42, va="center")
    else:
        ax.set_yticklabels([])

out_path_png = PROJECT_ROOT / "reports/figures/final_detail_analysis.png"
out_path_jpg = PROJECT_ROOT / "reports/figures/final_detail_analysis.jpg"
plt.savefig(out_path_png, dpi=300)
plt.savefig(out_path_jpg, dpi=300)

# Copy to reports/figs as well
figs_dir = PROJECT_ROOT / "reports/figs"
figs_dir.mkdir(parents=True, exist_ok=True)
import shutil
shutil.copy(out_path_png, figs_dir / "final_detail_analysis.png")
shutil.copy(out_path_jpg, figs_dir / "final_detail_analysis.jpg")

plt.close()
print(f"Unified detail figure successfully created at:\n{out_path_png}")
