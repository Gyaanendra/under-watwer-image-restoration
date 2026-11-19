import sys
from pathlib import Path
import time
import torch

PROJECT_ROOT = Path("E:/sem5_project")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/HeroMamba"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/waterMamba"))

from scripts.waterMamba.watermamba_model import WaterMamba
from scripts.waterMamba.core import build_model
from scripts.HeroMamba.heromamba_model import HeroMamba
from models_factory import build_experiment_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
x = torch.randn(1, 3, 256, 256, device=device)

models = {}
models["WM 1D (CUDA)"] = WaterMamba(dim=32, img_size=256, blocks=1, scan_mode="1d").to(device)
wm_base, _ = build_model(dim=32, img_size=256, blocks=1, checkpoint_path=None, device=device, scan_mode="auto")
models["WM 2D (Pooled)"] = wm_base.to(device)
models["HM 1D (CUDA)"] = HeroMamba(dims=(32, 64, 128, 256), scan_mode="1d").to(device)
models["HM 2D (Pooled)"] = HeroMamba(dims=(32, 64, 128, 256), scan_mode="auto").to(device)
models["W-1a (SCA)"] = build_experiment_model("wm", keep_ccoss=0, attn="sca", act="base").to(device)
models["H-1b (RC Gate)"] = build_experiment_model("hm", attn="rc", act="base", norm="bn", scales=0).to(device)
models["H-D3-Ada"] = build_experiment_model("hm", attn="none", act="adaptivegate", norm="ln", scales=1).to(device)

print(f"\n{'Model':<25} | {'Latency (ms)':<15} | {'FPS':<10}")
print("-" * 55)

for name, m in models.items():
    m.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = m(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        N = 30
        t0 = time.time()
        for _ in range(N):
            _ = m(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_ms = ((time.time() - t0) / N) * 1000.0
        fps = 1000.0 / lat_ms
    print(f"{name:<25} | {lat_ms:10.2f} ms | {fps:6.1f} FPS")
