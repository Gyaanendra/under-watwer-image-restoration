"""
Alias wrapper for watermamba_model.py to support alternative spelling / imports.
"""
from .watermamba_model import (
    selective_scan_ref,
    selective_scan,
    S6,
    SOSS,
    ChannelScan,
    CCOSS,
    MSFFN,
    SCOSSBlock,
    WaterMamba,
    get_model,
    HAS_MAMBA,
)

__all__ = [
    "selective_scan_ref",
    "selective_scan",
    "S6",
    "SOSS",
    "ChannelScan",
    "CCOSS",
    "MSFFN",
    "SCOSSBlock",
    "WaterMamba",
    "get_model",
    "HAS_MAMBA",
]

if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = WaterMamba(dim=16, img_size=32).to(device)
    dummy = torch.rand(1, 3, 32, 32, device=device)
    print("Alias test OK:", net(dummy).shape)
