"""
Hero-Mamba (arXiv: 2404.16266) - High-Efficiency and Robust Underwater Image Restoration
with State Space Models and Physical Priors.

Key Features:
1. Dual-domain Stem: Fuses spatial RGB with FFT frequency spectral domain (amplitude and phase).
2. SS2D: 4-directional 2D selective scan state space model.
3. MS-Fusion: Multi-scale parallel depthwise separable convolutions (1x1, 3x3, 5x5).
4. BackgroundLightPrior: Physics-informed illuminant estimate with learnable offsets.
5. ColorFusion: Physical atmospheric/optical model (J = I * t + B * (1 - t)) at feature level.
6. HeroLoss: Composite loss with L1, SSIM, and VGG-19 perceptual contrastive distance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False


# --------------------------------------------------------------------------- #
# 1. S6 / Selective Scan Core
# --------------------------------------------------------------------------- #
def selective_scan_ref(u, delta, A, B, C, D, delta_bias):
    b, d, l = u.shape
    g = B.shape[1]
    u, delta = u.float(), F.softplus(delta.float() + delta_bias[None, :, None])
    B = B.float().repeat_interleave(d // g, dim=1)
    C = C.float().repeat_interleave(d // g, dim=1)
    x = u.new_zeros(b, d, A.shape[1])
    ys = []
    for i in range(l):
        dt = delta[:, :, i, None]
        x = torch.exp(dt * A) * x + dt * B[:, :, :, i] * u[:, :, i, None]
        ys.append((x * C[:, :, :, i]).sum(-1))
    return torch.stack(ys, -1) + u * D[None, :, None]


def selective_scan(u, delta, A, B, C, D, delta_bias):
    if HAS_MAMBA and u.is_cuda:
        return selective_scan_fn(u, delta, A, B, C, D, z=None,
                                 delta_bias=delta_bias, delta_softplus=True)
    return selective_scan_ref(u, delta, A, B, C, D, delta_bias)


class S6(nn.Module):
    def __init__(self, d, d_state=16, K=1):
        super().__init__()
        self.d, self.N, self.K = d, d_state, K
        self.R = math.ceil(d / 16)
        self.x_proj_weight = nn.Parameter(torch.randn(K, self.R + 2 * d_state, d) * (d ** -0.5))
        self.dt_proj_weight = nn.Parameter(torch.empty(K, d, self.R).uniform_(-self.R ** -0.5, self.R ** -0.5))
        dt = torch.exp(torch.rand(K, d) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(K * d, 1))
        self.D = nn.Parameter(torch.ones(K * d))

    def forward(self, xs):
        b, K, d, L = xs.shape
        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.R, self.N, self.N], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_proj_weight)
        y = selective_scan(
            xs.reshape(b, K * d, L).contiguous(), dts.reshape(b, K * d, L).contiguous(),
            -torch.exp(self.A_log.float()), Bs.contiguous(), Cs.contiguous(),
            self.D.float(), self.dt_bias.reshape(-1).float())
        return y.view(b, K, d, L)


class SS2D(nn.Module):
    """
    Spatial Selective Scan supporting:
    - 2D Spatial Scan: Separable vertical (H) and horizontal (W) coordinate sweeps.
      Ultra-lightweight (~3 GB VRAM at 256x256), native Windows support without CUDA C++ compilers.
    - 1D Flattened Scan: Original 4-way unrolled scan (ideal with mamba_ssm CUDA kernel).
    """
    def __init__(self, dim, d_state=16, expand=2, scan_mode="auto"):
        super().__init__()
        di = dim * expand
        self.dim = dim
        self.di = di
        self.in_proj = nn.Linear(dim, di * 2)
        self.dw = nn.Conv2d(di, di, 3, padding=1, groups=di)
        self.scan_mode = scan_mode
        self.use_2d = (scan_mode == "2d") or (scan_mode == "auto" and not HAS_MAMBA)

        if self.use_2d:
            self.ssm_h = S6(di, d_state, K=2)
            self.ssm_w = S6(di, d_state, K=2)
        else:
            self.ssm = S6(di, d_state, K=4)

        self.norm = nn.LayerNorm(di)
        self.out_proj = nn.Linear(di, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        u, z = self.in_proj(x.permute(0, 2, 3, 1)).chunk(2, -1)
        u = F.silu(self.dw(u.permute(0, 3, 1, 2)))

        if self.use_2d:
            # 1) Vertical bidirectional scan along H
            yh = u.mean(-1).unsqueeze(1)                      # (B, 1, di, H)
            yh_bidir = torch.cat([yh, yh.flip(-1)], 1)       # (B, 2, di, H)
            zh = self.ssm_h(yh_bidir)
            zh = (zh[:, 0] + zh[:, 1].flip(-1)).unsqueeze(-1) # (B, di, H, 1)

            # 2) Horizontal bidirectional scan along W
            yw = u.mean(-2).unsqueeze(1)                      # (B, 1, di, W)
            yw_bidir = torch.cat([yw, yw.flip(-1)], 1)       # (B, 2, di, W)
            zw = self.ssm_w(yw_bidir)
            zw = (zw[:, 0] + zw[:, 1].flip(-1)).unsqueeze(-2) # (B, di, 1, W)

            y = u * torch.sigmoid(zh + zw)
            y = self.norm(y.permute(0, 2, 3, 1))
        else:
            L = H * W
            xh = torch.stack([u.flatten(2), u.transpose(2, 3).flatten(2)], 1)
            xs = torch.cat([xh, xh.flip(-1)], 1)
            y = self.ssm(xs)
            y = (y[:, 0] + y[:, 2].flip(-1)
                 + y[:, 1].reshape(B, -1, W, H).transpose(2, 3).reshape(B, -1, L)
                 + y[:, 3].flip(-1).reshape(B, -1, W, H).transpose(2, 3).reshape(B, -1, L))
            y = self.norm(y.view(B, -1, H, W).permute(0, 2, 3, 1))

        y = y * F.silu(z)
        return self.out_proj(y).permute(0, 3, 1, 2)


# --------------------------------------------------------------------------- #
# 2. MS-Fusion Block
# --------------------------------------------------------------------------- #
class MSFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        def branch(k):
            p = k // 2
            return nn.Sequential(
                nn.Conv2d(dim, dim, k, padding=p),
                nn.Conv2d(dim, dim, k, padding=p, groups=dim),
                nn.SiLU(),
                nn.Conv2d(dim, dim, k, padding=p),
                nn.Conv2d(dim, dim, k, padding=p, groups=dim))
        self.branches = nn.ModuleList([branch(1), branch(3), branch(5)])
        self.reduce = nn.Conv2d(3 * dim, dim, 1)

    def forward(self, x):
        return self.reduce(torch.cat([b(x) for b in self.branches], 1)) + x


# --------------------------------------------------------------------------- #
# 3. Background-Light Prior & ColorFusion
# --------------------------------------------------------------------------- #
class BackgroundLightPrior(nn.Module):
    def __init__(self, chans, fc4=None):
        super().__init__()
        self.fc4 = fc4
        if fc4 is not None:
            for p in fc4.parameters():
                p.requires_grad = False
            fc4.eval()
        self.b_l = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.convs = nn.ModuleList([nn.Conv2d(3, c, 1) for c in chans])

    @torch.no_grad()
    def estimate(self, img):
        e = self.fc4(img) if self.fc4 is not None else img.mean((2, 3))
        e = e / (e.sum(1, keepdim=True) + 1e-6)
        return e.view(-1, 3, 1, 1)

    def forward(self, img, sizes):
        base = self.estimate(img) + self.b_l
        return [conv(base.expand(-1, -1, h, w)) for conv, (h, w) in zip(self.convs, sizes)]


class ColorFusion(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.to_bf = nn.Conv2d(c, c, 1)
        self.to_t = nn.Conv2d(c, c, 1)
        self.to_w = nn.Conv2d(2 * c, c, 1)

    def forward(self, f, bp):
        bf = torch.sigmoid(self.to_bf(f))
        w = torch.sigmoid(self.to_w(torch.cat([bf, bp], 1)))
        B = w * bf + (1 - w) * bp
        t = torch.sigmoid(self.to_t(f))
        return f * t + B * (1 - t)


# --------------------------------------------------------------------------- #
# 4. Encoder & Decoder Blocks
# --------------------------------------------------------------------------- #
class EncoderBlock(nn.Module):
    def __init__(self, cin, cout, scan_mode="auto"):
        super().__init__()
        self.norm = nn.GroupNorm(1, cin)
        self.ss2d = SS2D(cin, scan_mode=scan_mode)
        self.msf = MSFusion(cin)
        self.down = nn.Conv2d(cin, cout, 2, stride=2)

    def forward(self, x):
        x = x + self.ss2d(self.norm(x))
        return self.down(self.msf(x))


class DecoderBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return F.silu(self.conv2(self.bn(self.conv1(x))))


# --------------------------------------------------------------------------- #
# 5. HeroMamba Network
# --------------------------------------------------------------------------- #
class HeroMamba(nn.Module):
    def __init__(self, fc4=None, dims=(32, 64, 128, 256), scan_mode="auto"):
        super().__init__()
        self.scan_mode = scan_mode
        dec = (dims[2], dims[1], dims[0], dims[0] // 2)

        # Dual-domain stem: RGB (3) + FFT amplitude/phase (2)
        self.ss2d_rgb = SS2D(3, scan_mode=scan_mode)
        self.ss2d_fft = SS2D(2, scan_mode=scan_mode)
        self.stem_msf = MSFusion(5)
        self.stem_down = nn.Conv2d(5, dims[0], 3, stride=2, padding=1)

        self.enc = nn.ModuleList([EncoderBlock(dims[i], dims[i + 1], scan_mode=scan_mode) for i in range(3)])
        self.prior = BackgroundLightPrior(dims, fc4)
        self.cf = nn.ModuleList([ColorFusion(c) for c in dims])

        ins = [dims[3] + dims[3], dec[0] + dims[2], dec[1] + dims[1], dec[2] + dims[0]]
        self.dec = nn.ModuleList([DecoderBlock(i, o) for i, o in zip(ins, dec)])
        self.out = nn.Conv2d(dec[3], 3, 3, padding=1)

    @staticmethod
    def spectral(x):
        spec = torch.fft.fft2(x.float().mean(1, keepdim=True), norm="ortho")
        return torch.cat([torch.log1p(spec.abs()), spec.angle()], 1)

    def forward(self, x):
        i_f = torch.cat([self.ss2d_rgb(x), self.ss2d_fft(self.spectral(x))], 1)
        f = self.stem_down(self.stem_msf(i_f))
        feats = [f]
        for blk in self.enc:
            f = blk(f)
            feats.append(f)
        bps = self.prior(x, [t.shape[-2:] for t in feats])
        cs = [cf(fi, bp) for cf, fi, bp in zip(self.cf, feats, bps)]
        d = feats[3]
        for i, blk in enumerate(self.dec):
            d = blk(torch.cat([d, cs[3 - i]], 1))
        return self.out(d) + x


# --------------------------------------------------------------------------- #
# 6. Composite Loss Function (HeroLoss)
# --------------------------------------------------------------------------- #
def _gauss_window(size=11, sigma=1.5, ch=3, device="cpu"):
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    return w.expand(ch, 1, size, size).contiguous()


def ssim(x, y, win=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    ch = x.shape[1]
    w = _gauss_window(win, sigma, ch, x.device).to(x.dtype)
    mu1 = F.conv2d(x, w, padding=win // 2, groups=ch)
    mu2 = F.conv2d(y, w, padding=win // 2, groups=ch)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(x * x, w, padding=win // 2, groups=ch) - mu1_sq
    sigma2_sq = F.conv2d(y * y, w, padding=win // 2, groups=ch) - mu2_sq
    sigma12 = F.conv2d(x * y, w, padding=win // 2, groups=ch) - mu1_mu2
    s = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return s.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision
        try:
            weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            feats = torchvision.models.vgg19(weights=weights).features.eval()
        except Exception:
            feats = torchvision.models.vgg19(pretrained=pretrained).features.eval()
        cuts = [(0, 2), (2, 7), (7, 12), (12, 21), (21, 30)]
        self.slices = nn.ModuleList([feats[a:b] for a, b in cuts])
        for p in self.parameters():
            p.requires_grad = False
        self.weights = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _feats(self, x):
        x, out = (x - self.mean) / self.std, []
        for s in self.slices:
            x = s(x)
            out.append(x)
        return out

    def forward(self, anchor, pos, neg):
        fa = self._feats(anchor)
        with torch.no_grad():
            fp, fn = self._feats(pos), self._feats(neg)
        return sum(w * F.l1_loss(a, p) / (F.l1_loss(a, n) + 1e-7)
                   for w, a, p, n in zip(self.weights, fa, fp, fn))


class HeroLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.8, gamma=0.1, pretrained_vgg=True):
        super().__init__()
        self.a, self.b, self.g = alpha, beta, gamma
        self.contrast = ContrastiveLoss(pretrained_vgg) if gamma > 0 else None

    def forward(self, pred, gt, raw):
        loss = self.a * F.l1_loss(pred, gt) + self.b * (1 - ssim(pred, gt))
        if self.contrast is not None:
            loss = loss + self.g * self.contrast(pred, gt, raw)
        return loss
