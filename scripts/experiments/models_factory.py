"""
Parametric Model Factory for Experimental Post-Baseline Sweeps.
Supports:
- WaterMamba with modular attention, activation, CCOSS retention, and scan variants.
- Hero-Mamba with modular attention, activation, decoder norm, residual scales, and scan variants.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.waterMamba.watermamba_model import S6, SOSS, CCOSS, selective_scan
from scripts.HeroMamba.heromamba_model import SS2D, BackgroundLightPrior, ColorFusion
from scripts.experiments.modules import (
    SimplifiedChannelAttention,
    RowColSpatialGate,
    SpatialAndChannelAttention,
    SimpleGate,
    AdaptiveGate,
    LayerNorm2d,
    ResidualScale,
    Pooled1DConvScan,
)


# =========================================================================== #
# 1. Configurable Feed-Forward Networks (MSFFN & MS-Fusion)
# =========================================================================== #
class ConfigurableMSFFN(nn.Module):
    """
    MSFFN supporting 'base' (ReLU), 'simplegate' (NAFNet), and 'adaptivegate'.
    """
    def __init__(self, dim, act="base"):
        super().__init__()
        self.act = act

        def make_branch(k):
            p = k // 2
            if act == "simplegate":
                # Expand to 2*dim, apply SimpleGate to halve back to dim
                return nn.Sequential(
                    nn.Conv2d(dim, dim, k, padding=p),
                    nn.Conv2d(dim, 2 * dim, k, padding=p, groups=1),
                    SimpleGate(),
                    nn.Conv2d(dim, dim, k, padding=p),
                    nn.Conv2d(dim, dim, k, padding=p, groups=dim)
                )
            elif act == "adaptivegate":
                # Expand to 2*dim, apply AdaptiveGate
                return nn.Sequential(
                    nn.Conv2d(dim, dim, k, padding=p),
                    nn.Conv2d(dim, 2 * dim, k, padding=p, groups=1),
                    AdaptiveGate(dim),
                    nn.Conv2d(dim, dim, k, padding=p),
                    nn.Conv2d(dim, dim, k, padding=p, groups=dim)
                )
            else:
                # Base: standard ReLU
                return nn.Sequential(
                    nn.Conv2d(dim, dim, k, padding=p),
                    nn.Conv2d(dim, dim, k, padding=p, groups=dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim, dim, k, padding=p),
                    nn.Conv2d(dim, dim, k, padding=p, groups=dim)
                )

        self.b1 = make_branch(1)
        self.b3 = make_branch(3)
        self.b5 = make_branch(5)
        self.merge = nn.Conv2d(3 * dim, dim, kernel_size=1)

    def forward(self, x):
        return self.merge(torch.cat([self.b1(x), self.b3(x), self.b5(x)], dim=1))


class ConfigurableMSFusion(nn.Module):
    """
    Hero-Mamba MS-Fusion supporting 'base' (SiLU), 'simplegate', and 'adaptivegate'.
    """
    def __init__(self, dim, act="base"):
        super().__init__()
        self.act = act

        def make_branch(k):
            p = k // 2
            if act == "simplegate":
                return nn.Sequential(
                    nn.Conv2d(dim, dim, k, padding=p),
                    nn.Conv2d(dim, 2 * dim, k, padding=p, groups=1),
                    SimpleGate(),
                    nn.Conv2d(dim, dim, k, padding=p),
                    nn.Conv2d(dim, dim, k, padding=p, groups=dim)
                )
            elif act == "adaptivegate":
                return nn.Sequential(
                    nn.Conv2d(dim, dim, k, padding=p),
                    nn.Conv2d(dim, 2 * dim, k, padding=p, groups=1),
                    AdaptiveGate(dim),
                    nn.Conv2d(dim, dim, k, padding=p),
                    nn.Conv2d(dim, dim, k, padding=p, groups=dim)
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(dim, dim, k, padding=p),
                    nn.Conv2d(dim, dim, k, padding=p, groups=dim),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(dim, dim, k, padding=p),
                    nn.Conv2d(dim, dim, k, padding=p, groups=dim)
                )

        self.branches = nn.ModuleList([make_branch(1), make_branch(3), make_branch(5)])
        self.reduce = nn.Conv2d(3 * dim, dim, kernel_size=1)

    def forward(self, x):
        return self.reduce(torch.cat([b(x) for b in self.branches], dim=1)) + x


# =========================================================================== #
# 2. Configurable WaterMamba Block & Network
# =========================================================================== #
class ConfigurableSCOSSBlock(nn.Module):
    def __init__(self, dim, size, keep_ccoss=1, attn="none", act="base", scan="pooled_ssm"):
        super().__init__()
        h, w = size
        self.n1 = nn.GroupNorm(1, dim)
        self.n2 = nn.GroupNorm(1, dim)

        # 1. Spatial Scan
        if scan == "pooled_conv":
            self.spatial_scan = Pooled1DConvScan(dim)
        elif scan == "full":
            self.spatial_scan = SOSS(dim, scan_mode="1d")
        else:
            self.spatial_scan = SOSS(dim, scan_mode="2d")

        # 2. Channel / Attention slot
        self.keep_ccoss = keep_ccoss
        if keep_ccoss == 1:
            self.channel_block = CCOSS(dim, h, w)
        else:
            if attn == "sca":
                self.channel_block = SimplifiedChannelAttention(dim)
            elif attn == "rc":
                self.channel_block = RowColSpatialGate(dim)
            elif attn == "rc_sca":
                self.channel_block = SpatialAndChannelAttention(dim)
            else:
                self.channel_block = nn.Identity()

        # 3. Feed-Forward Network
        self.msffn = ConfigurableMSFFN(dim, act=act)

    def forward(self, x):
        x = x + self.channel_block(self.spatial_scan(self.n1(x)))
        x = x + self.msffn(self.n2(x))
        return x


class ExperimentalWaterMamba(nn.Module):
    def __init__(self, dim=32, img_size=128, keep_ccoss=1, attn="none", act="base", scan="pooled_ssm"):
        super().__init__()
        dims = [dim, dim * 2, dim * 4, dim * 8]
        sizes = [img_size // (2 ** i) for i in range(4)]

        self.head = nn.Conv2d(3, dims[0], kernel_size=3, padding=1)

        # Encoder (4 levels)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(4):
            blk = ConfigurableSCOSSBlock(dims[i], (sizes[i], sizes[i]),
                                        keep_ccoss=keep_ccoss, attn=attn, act=act, scan=scan)
            self.enc.append(blk)
            if i < 3:
                self.down.append(nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))

        # Bottleneck
        self.bot = ConfigurableSCOSSBlock(dims[3], (sizes[3], sizes[3]),
                                         keep_ccoss=keep_ccoss, attn=attn, act=act, scan=scan)

        # Decoder (4 levels)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(3, 0, -1):
            self.up.append(nn.ConvTranspose2d(dims[i], dims[i - 1], kernel_size=2, stride=2))
            blk = ConfigurableSCOSSBlock(dims[i - 1], (sizes[i - 1], sizes[i - 1]),
                                        keep_ccoss=keep_ccoss, attn=attn, act=act, scan=scan)
            self.dec.append(blk)

        self.tail = nn.Conv2d(dims[0], 3, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.head(x)
        skips = []
        for i in range(3):
            h = self.enc[i](h)
            skips.append(h)
            h = self.down[i](h)

        h = self.enc[3](h)
        h = self.bot(h)

        for i in range(3):
            h = self.up[i](h) + skips[2 - i]
            h = self.dec[i](h)

        return self.tail(h) + x


# =========================================================================== #
# 3. Configurable Hero-Mamba Blocks & Network
# =========================================================================== #
class ConfigurableHeroEncoderBlock(nn.Module):
    def __init__(self, cin, cout, attn="none", act="base", scan="pooled_ssm"):
        super().__init__()
        self.norm = nn.GroupNorm(1, cin)

        # 1. Scan
        if scan == "pooled_conv":
            self.scan = Pooled1DConvScan(cin)
        elif scan == "full":
            self.scan = SS2D(cin, scan_mode="1d")
        else:
            self.scan = SS2D(cin, scan_mode="2d")

        # 2. Attention
        if attn == "sca":
            self.attn = SimplifiedChannelAttention(cin)
        elif attn == "rc":
            self.attn = RowColSpatialGate(cin)
        elif attn == "rc_sca":
            self.attn = SpatialAndChannelAttention(cin)
        else:
            self.attn = nn.Identity()

        # 3. MS-Fusion
        self.msf = ConfigurableMSFusion(cin, act=act)
        self.down = nn.Conv2d(cin, cout, kernel_size=2, stride=2)

    def forward(self, x):
        scanned = self.scan(self.norm(x))
        attended = self.attn(scanned)
        x = x + attended
        return self.down(self.msf(x))


class ConfigurableHeroDecoderBlock(nn.Module):
    def __init__(self, cin, cout, norm="bn", act="base", scales=0):
        super().__init__()
        self.scales = scales
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, padding=1)
        self.norm = LayerNorm2d(cout) if norm == "ln" else nn.BatchNorm2d(cout)

        if act == "simplegate":
            self.conv2 = nn.Conv2d(cout, 2 * cout, kernel_size=3, padding=1)
            self.act_layer = SimpleGate()
        elif act == "adaptivegate":
            self.conv2 = nn.Conv2d(cout, 2 * cout, kernel_size=3, padding=1)
            self.act_layer = AdaptiveGate(cout)
        else:
            self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, padding=1)
            self.act_layer = nn.SiLU(inplace=True)

        if scales == 1:
            self.res_scale = ResidualScale(cout)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        fx = self.conv1(x)
        fx = self.norm(fx)
        fx = self.act_layer(self.conv2(fx))

        if self.scales == 1:
            return self.res_scale(x[:, :fx.shape[1]], fx)
        return fx


class ExperimentalHeroMamba(nn.Module):
    def __init__(self, dims=(32, 64, 128, 256), attn="none", act="base",
                 norm="bn", scales=0, scan="pooled_ssm"):
        super().__init__()
        dec = (dims[2], dims[1], dims[0], dims[0] // 2)

        # Dual-domain stem: RGB (3) + FFT amplitude/phase (2)
        scan_mode = "1d" if scan == "full" else "2d"
        self.ss2d_rgb = SS2D(3, scan_mode=scan_mode)
        self.ss2d_fft = SS2D(2, scan_mode=scan_mode)
        self.stem_msf = ConfigurableMSFusion(5, act=act)
        self.stem_down = nn.Conv2d(5, dims[0], kernel_size=3, stride=2, padding=1)

        # Encoder (3 blocks)
        self.enc = nn.ModuleList([
            ConfigurableHeroEncoderBlock(dims[i], dims[i + 1], attn=attn, act=act, scan=scan)
            for i in range(3)
        ])

        # Physical Optical Prior
        self.prior = BackgroundLightPrior(dims)
        self.cf = nn.ModuleList([ColorFusion(c) for c in dims])

        # Decoder (4 blocks)
        ins = [dims[3] + dims[3], dec[0] + dims[2], dec[1] + dims[1], dec[2] + dims[0]]
        self.dec = nn.ModuleList([
            ConfigurableHeroDecoderBlock(ins[i], dec[i], norm=norm, act=act, scales=scales)
            for i in range(4)
        ])
        self.out = nn.Conv2d(dec[3], 3, kernel_size=3, padding=1)

    @staticmethod
    def spectral(x):
        spec = torch.fft.fft2(x.float().mean(1, keepdim=True), norm="ortho")
        return torch.cat([torch.log1p(spec.abs()), spec.angle()], dim=1)

    def forward(self, x):
        i_f = torch.cat([self.ss2d_rgb(x), self.ss2d_fft(self.spectral(x))], dim=1)
        f = self.stem_down(self.stem_msf(i_f))
        feats = [f]
        for blk in self.enc:
            f = blk(f)
            feats.append(f)

        bps = self.prior(x, [t.shape[-2:] for t in feats])
        cs = [cf(fi, bp) for cf, fi, bp in zip(self.cf, feats, bps)]

        d = feats[3]
        for i, blk in enumerate(self.dec):
            d = blk(torch.cat([d, cs[3 - i]], dim=1))

        return self.out(d) + x


# =========================================================================== #
# 4. Master Model Builder
# =========================================================================== #
def build_experiment_model(model_type="wm", keep_ccoss=1, attn="none", act="base",
                           norm="bn", scales=0, scan="pooled_ssm", img_size=128):
    """
    Factory creating configured model instance according to experimental run flags.
    """
    if model_type.lower() in ["wm", "watermamba"]:
        model = ExperimentalWaterMamba(
            dim=32,
            img_size=img_size,
            keep_ccoss=keep_ccoss,
            attn=attn,
            act=act,
            scan=scan
        )
    elif model_type.lower() in ["hm", "heromamba", "hero-mamba"]:
        model = ExperimentalHeroMamba(
            dims=(32, 64, 128, 256),
            attn=attn,
            act=act,
            norm=norm,
            scales=scales,
            scan=scan
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'wm' or 'hm'.")

    return model
