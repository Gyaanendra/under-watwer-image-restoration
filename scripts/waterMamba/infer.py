"""
WaterMamba Single-Image / Folder Inference Pipeline.
Supports enhancing images of ANY original size and aspect ratio using:
1. 'guided' / 'resize' mode: Runs network at 256x256 and applies color/illumination transfer
   to preserve 100% of the original full-resolution sharpness and detail.
2. 'tile' mode: Sliding window overlapping 256x256 patches blended seamlessly with Gaussian feathering.
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

# Ensure project root in path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.waterMamba.watermamba_model import WaterMamba
from scripts.waterMamba.core import build_model


def enhance_guided(net, img_pil, device):
    """
    Enhances an image of arbitrary size by computing residual color/illumination correction
    at 256x256 and transferring it to the full-resolution original image.
    Preserves original native resolution and edge sharpness.
    """
    orig_w, orig_h = img_pil.size

    # Resize to model training size 256x256
    img_low = TF.resize(img_pil, [256, 256])
    t_low = TF.to_tensor(img_low).unsqueeze(0).to(device)

    with torch.no_grad():
        t_enh_low = net(t_low).clamp(0.0, 1.0)

    # Compute color illumination ratio / residual
    ratio_low = (t_enh_low + 1e-4) / (t_low + 1e-4)

    # Upsample the illumination transform back to original resolution with smooth interpolation
    ratio_high = F.interpolate(ratio_low, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

    t_orig = TF.to_tensor(img_pil).unsqueeze(0).to(device)
    enhanced_high = (t_orig * ratio_high).clamp(0.0, 1.0)

    # Convert back to PIL Image
    enhanced_np = (enhanced_high.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(enhanced_np)


def enhance_tiled(net, img_pil, device, tile_size=256, overlap=32):
    """
    Enhances an arbitrary size image by dividing it into overlapping 256x256 tiles,
    processing each tile, and blending them with a cosine window to prevent border artifacts.
    """
    orig_w, orig_h = img_pil.size
    t_orig = TF.to_tensor(img_pil).unsqueeze(0).to(device)

    stride = tile_size - overlap
    pad_h = (stride - (orig_h - tile_size) % stride) % stride if orig_h > tile_size else (tile_size - orig_h)
    pad_w = (stride - (orig_w - tile_size) % stride) % stride if orig_w > tile_size else (tile_size - orig_w)

    padded = F.pad(t_orig, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, H_pad, W_pad = padded.shape

    out = torch.zeros_like(padded)
    weight_map = torch.zeros_like(padded)

    # Create 2D blending window
    y_win = torch.sin(torch.linspace(0, torch.pi, tile_size, device=device))
    x_win = torch.sin(torch.linspace(0, torch.pi, tile_size, device=device))
    window = (y_win[:, None] @ x_win[None, :]).unsqueeze(0).unsqueeze(0).clamp(min=1e-3)

    for y in range(0, H_pad - tile_size + 1, stride):
        for x in range(0, W_pad - tile_size + 1, stride):
            tile = padded[:, :, y : y + tile_size, x : x + tile_size]
            with torch.no_grad():
                enh_tile = net(tile).clamp(0.0, 1.0)

            out[:, :, y : y + tile_size, x : x + tile_size] += enh_tile * window
            weight_map[:, :, y : y + tile_size, x : x + tile_size] += window

    result = (out / weight_map.clamp(min=1e-5))[:, :, :orig_h, :orig_w].clamp(0.0, 1.0)
    enhanced_np = (result.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(enhanced_np)


def infer_image(net, image_path, output_dir, device, mode="guided"):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    if mode == "tiled":
        enhanced = enhance_tiled(net, img, device)
    else:
        enhanced = enhance_guided(net, img, device)

    fname = Path(image_path).stem
    out_path = os.path.join(output_dir, f"{fname}_enhanced.png")
    enhanced.save(out_path)

    # Also save a side-by-side comparison at original aspect ratio
    comp_w = orig_w * 2
    comp_img = Image.new("RGB", (comp_w, orig_h))
    comp_img.paste(img, (0, 0))
    comp_img.paste(enhanced, (orig_w, 0))
    comp_path = os.path.join(output_dir, f"{fname}_comparison.png")
    comp_img.save(comp_path)

    return out_path, comp_path, (orig_w, orig_h)


def parse_arguments():
    parser = argparse.ArgumentParser(description="WaterMamba Original-Resolution Inference")

    parser.add_argument("--input", type=str, required=True, help="Path to single image file or directory of images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/watermamba/watermamba_best.pth", help="Model weights path")
    parser.add_argument("--output_dir", type=str, default="results/watermamba/inferences", help="Directory to save enhanced images")
    parser.add_argument("--mode", type=str, choices=["guided", "tiled"], default="guided",
                        help="'guided' (fast high-res detail reconstruction) or 'tiled' (sliding window patches)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    device_name = "cpu" if args.cpu else "cuda"
    net, device = build_model(
        dim=32,
        img_size=256,
        blocks=1,
        checkpoint_path=args.checkpoint,
        device=device_name,
        scan_mode="auto"
    )
    net.eval()

    # Collect images
    if os.path.isfile(args.input):
        image_files = [args.input]
    elif os.path.isdir(args.input):
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        image_files = []
        for ext in valid_exts:
            image_files.extend(glob.glob(os.path.join(args.input, ext)))
            image_files.extend(glob.glob(os.path.join(args.input, ext.upper())))
        image_files = sorted(list(set(image_files)))
    else:
        raise FileNotFoundError(f"Input path '{args.input}' not found.")

    print("\n" + "=" * 80)
    print(" WATERMAMBA ORIGINAL-RESOLUTION INFERENCE")
    print(f" Images Found:   {len(image_files)}")
    print(f" Mode:           {args.mode.upper()}")
    print(f" Checkpoint:     {args.checkpoint}")
    print(f" Output Folder:  {os.path.abspath(args.output_dir)}")
    print("=" * 80)

    for idx, img_p in enumerate(image_files, 1):
        out_p, comp_p, (w, h) = infer_image(net, img_p, args.output_dir, device, mode=args.mode)
        print(f"[{idx:03d}/{len(image_files):03d}] Enhanced {Path(img_p).name} ({w}x{h}) -> {out_p}")

    print("\n[Done] All images processed successfully!")
    print(f"Check results at: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
