"""
Modular Neural Components for Post-Baseline Experiments.
Implements:
1. Simplified Channel Attention (SCA)
2. Row/Column Spatial Gate (RowColSpatialGate)
3. Simple Gate (NAFNet)
4. Adaptive Gate (AdaGate with channel scale + context modulation)
5. Pooled 1D Convolution Block (SSM-free ablation)
6. Residual Scale Block (learnable beta, gamma)
7. LayerNorm2d for CNN/Mamba feature maps
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.waterMamba.watermamba_model import S6


# --------------------------------------------------------------------------- #
# 1. Attention Modules (Group A)
# --------------------------------------------------------------------------- #
class SimplifiedChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA):
    Computes global average pooling per channel, projects via 1x1 Conv,
    and scales input feature maps.
    """
    def __init__(self, dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (B, C, H, W)
        scale = self.conv(self.gap(x))
        return x * scale


class RowColSpatialGate(nn.Module):
    """
    Row/Column Spatial Gate:
    Averages over width to scan along H, and averages over height to scan along W,
    using bidirectional S6 state-space models. Combines into a 2D spatial gate.
    """
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.ssm_h = S6(dim, d_state=d_state, K=2)
        self.ssm_w = S6(dim, d_state=d_state, K=2)

    def forward(self, x):
        B, C, H, W = x.shape
        # Vertical scan along H (mean over W)
        yh = x.mean(-1).unsqueeze(1)                        # (B, 1, C, H)
        yh_bidir = torch.cat([yh, yh.flip(-1)], dim=1)      # (B, 2, C, H)
        zh = self.ssm_h(yh_bidir)
        zh = (zh[:, 0] + zh[:, 1].flip(-1)).unsqueeze(-1)   # (B, C, H, 1)

        # Horizontal scan along W (mean over H)
        yw = x.mean(-2).unsqueeze(1)                        # (B, 1, C, W)
        yw_bidir = torch.cat([yw, yw.flip(-1)], dim=1)      # (B, 2, C, W)
        zw = self.ssm_w(yw_bidir)
        zw = (zw[:, 0] + zw[:, 1].flip(-1)).unsqueeze(-2)   # (B, C, 1, W)

        # Spatial gating modulation
        gate = torch.sigmoid(zh + zw)
        return x * gate


class SpatialAndChannelAttention(nn.Module):
    """
    Sequential combination: Row/Col Spatial Gate followed by SCA.
    """
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.spatial_gate = RowColSpatialGate(dim, d_state=d_state)
        self.channel_attn = SimplifiedChannelAttention(dim)

    def forward(self, x):
        return self.channel_attn(self.spatial_gate(x))


# --------------------------------------------------------------------------- #
# 2. Gating & Activation Modules (Group B)
# --------------------------------------------------------------------------- #
class SimpleGate(nn.Module):
    """
    NAFNet Simple Gate: Splits 2C channels into two halves and multiplies them.
    Replaces activation functions without saturation or parameters.
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class AdaptiveGate(nn.Module):
    """
    Adaptive Gate (AdaGate):
    Splits 2C channels into (x1, x2). Gating is dynamically modulated by:
    1. Learnable per-channel affine parameters (weight & bias).
    2. Global scene context squeeze-and-excitation soft-gate.
    """
    def __init__(self, dim):
        super().__init__()
        # Learnable channel scale and bias
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        # Scene context modulation
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        context = torch.sigmoid(self.fc(self.gap(x2)))
        gate = (self.weight * x2 + self.bias) * context
        return x1 * gate


# --------------------------------------------------------------------------- #
# 3. Normalization & Residual Scales
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.Module):
    """
    Channels-first LayerNorm for (B, C, H, W) feature maps.
    Uses GroupNorm(1, C) for exact mathematical equivalence.
    """
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(1, num_channels, eps=eps)

    def forward(self, x):
        return self.gn(x)


class ResidualScale(nn.Module):
    """
    NAFNet learnable residual scaling.
    Initializes beta=0 and gamma=1 so the block starts as identity:
    y = gamma * x + beta * f(x)
    """
    def __init__(self, dim):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x, fx):
        return self.gamma * x + self.beta * fx


# --------------------------------------------------------------------------- #
# 4. SSM-Free Ablation (Group D)
# --------------------------------------------------------------------------- #
class Pooled1DConvScan(nn.Module):
    """
    Ablation W-R2 / H-R2:
    Keeps row and column average pooling, but replaces the S6 state space
    recurrent scan with lightweight 1D depthwise convolutions.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv_h = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv_w = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        yh = self.conv_h(x.mean(-1)).unsqueeze(-1)    # (B, C, H, 1)
        yw = self.conv_w(x.mean(-2)).unsqueeze(-2)    # (B, C, 1, W)
        return x * torch.sigmoid(yh + yw)
