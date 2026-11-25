"""
Torch-side machinery for the count-loss project: v2 dataset, counter
dataset, the small U-Net counter, the count loss and the density-stratified
sampler. Kept separate from countlib.py so CPU StarDist workers never import
torch.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import countlib as cl

v1_common, v1_models = cl.import_v1()


# =============================================================================
# V2 QUASI-PAIRED DATASET (brief option ii)
# =============================================================================

class V2PatchDataset(Dataset):
    """Quasi-paired v2 patches: one (y0, x0, size) slice of the pre-warped
    arrays gives SRS and H&E at the same tissue. NO flip anywhere.

    rows needs: core, y0, x0, size, render, patch_uid, confidence_weight,
    n_real (StarDist count on the real H&E; NaN where not cached).
    """

    def __init__(self, rows: pd.DataFrame, config: dict, calibration):
        if config.get("SRS_CORE_FLIP_HORIZONTAL", False) or config.get("HE_CORE_FLIP_HORIZONTAL", False):
            raise ValueError(
                "SRS_CORE_FLIP_HORIZONTAL / HE_CORE_FLIP_HORIZONTAL must be False on the v2 data path: "
                "v2's warped H&E already carries the mirror. Setting either double-flips the pair "
                "(see DECISION_LOG D-01/D-02)."
            )
        self.rows = rows.reset_index(drop=True)
        self.config = config
        self.calibration = calibration
        self.channels = list(config["SRS_CHANNEL_INDICES"])
        self.norm_mode = config["SRS_NORM_MODE"]
        self.size = int(config["NETWORK_INPUT_SIZE"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        srs, he = cl.v2_patch_native(r.core, int(r.y0), int(r.x0), int(r["size"]), channels=self.channels)
        srs512 = cl.resize_chw(srs, self.size)
        srs_n = cl.normalize_srs(srs512, r.core, self.channels, self.norm_mode,
                                 self.config.get("TYPE_CHANNEL_NORM", "global"))
        he_n = v1_common.rgb_to_he(cl.resize_rgb(he, self.size), self.calibration)
        extra = {}
        if self.config.get("SRS_EMPTY_LOSS", False):
            s12, _ = cl.v2_patch_native(r.core, int(r.y0), int(r.x0), int(r["size"]), channels=[1, 2])
            extra["srs_empty"] = torch.from_numpy(cl.srs_empty_mask(r.core, s12, self.size))[None]
        return dict(**extra,
            srs=torch.from_numpy(srs_n),
            he=torch.from_numpy(he_n),
            confidence_weight=torch.tensor(float(r.confidence_weight), dtype=torch.float32),
            n_real=torch.tensor(float(r.n_real) if pd.notna(r.n_real) else -1.0, dtype=torch.float32),
            patch_uid=r.patch_uid, core=r.core,
        )


def confidence_weights(rows: pd.DataFrame) -> pd.Series:
    """v2 per-sample confidence for the quasi-paired terms (DECISION_LOG D-05).
    v1's weight combined patch NMI / edge correlation / tissue fraction / core
    score; v2 provides no patch-level alignment metric. Registered cores
    (tiers A and B, groups rotating / legacy_test) get 1.0 scaled by valid
    area fraction; unpaired_only cores get 0 so the quasi-paired terms are
    masked for them. Tier B is NOT down-weighted: it is correctly registered,
    and weighting on r_50um would quietly gate stroma-rich G-1."""
    w = rows["valid_frac"].astype(float).clip(0, 1)
    w[rows["group"] == "unpaired_only"] = 0.0
    return w


# =============================================================================
# COUNTER DATASET + AUGMENTATION
# =============================================================================

class CounterDataset(Dataset):
    """Real H channel (calibrated [-1, 1], exactly as rgb_to_he produces it)
    plus the StarDist target heatmap (integral == count)."""

    def __init__(self, rows: pd.DataFrame, calibration, sigma: float, augment: bool, size: int = cl.RENDER):
        self.rows = rows.reset_index(drop=True)
        self.calibration = calibration
        self.sigma = float(sigma)
        self.augment = augment
        self.size = size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        from scipy.ndimage import gaussian_filter
        r = self.rows.iloc[idx]
        _, he = cl.v2_patch_native(r.core, int(r.y0), int(r.x0), int(r["size"]), channels=[2])
        h = v1_common.rgb_to_he(cl.resize_rgb(he, self.size), self.calibration)[0]
        cy, cx, _ = cl.stardist_cache()[r.patch_uid]
        hm = cl.heatmap_from_centroids(cy, cx, self.sigma, self.size)
        if self.augment:
            rng = np.random.default_rng()
            k = int(rng.integers(4))
            h = np.rot90(h, k); hm = np.rot90(hm, k)
            if rng.random() < 0.5:
                h = h[:, ::-1]; hm = hm[:, ::-1]
            s = rng.uniform(0.0, 1.5)
            if s > 0.1:
                h = gaussian_filter(h, s)
            # intensity scale +/-10% on stain amount (h + 1 is 0 for no stain)
            h = (h + 1.0) * rng.uniform(0.9, 1.1) - 1.0
            h = h + rng.normal(0.0, rng.uniform(0.0, 0.04), h.shape)
            h = np.clip(h, -1.0, 1.0)
        return dict(h=torch.from_numpy(np.ascontiguousarray(h, dtype=np.float32))[None],
                    hm=torch.from_numpy(np.ascontiguousarray(hm, dtype=np.float32))[None],
                    count=torch.tensor(float(len(cy)), dtype=torch.float32),
                    patch_uid=r.patch_uid)


# =============================================================================
# COUNTER NETWORK
# =============================================================================

def _block(cin, cout, dilation=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNetCounter(nn.Module):
    """3-level U-Net, 32 base filters, softplus output. Input (B, 1, H, W)
    calibrated H in [-1, 1]. Output (B, 1, H, W) density in nuclei per pixel,
    so .sum() over a region is a count.

    BatchNorm (fixed statistics at eval) rather than Instance/GroupNorm:
    per-sample normalisation would make the counter blind to absolute stain
    level (v1 FINDINGS 4b). The bottleneck uses dilated convs to reach a
    receptive field (~90 px) wider than a pair of touching nuclei.
    """

    DENSITY_SCALE = 1000.0  # network predicts density x SCALE (well-scaled regression target)

    def __init__(self, base: int = 32):
        super().__init__()
        self.e1 = _block(1, base)
        self.e2 = _block(base, base * 2)
        self.e3 = nn.Sequential(_block(base * 2, base * 4), _block(base * 4, base * 4, dilation=2))
        self.u2 = nn.Conv2d(base * 4, base * 2, 1)
        self.d2 = _block(base * 4, base * 2)
        self.u1 = nn.Conv2d(base * 2, base, 1)
        self.d1 = _block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward_scaled(self, x):
        e1 = self.e1(x)
        # avg (not max) pooling: max_pool2d's CUDA backward is not
        # deterministic, and the generator's gradient flows through here
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(self.u2(e3), scale_factor=2, mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(self.u1(d2), scale_factor=2, mode="nearest"), e1], 1))
        return F.softplus(self.out(d1))

    def forward(self, x):
        return self.forward_scaled(x) / self.DENSITY_SCALE


def load_counter(path: str, device) -> UNetCounter:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = UNetCounter(base=ck.get("base", 32)).to(device)
    net.load_state_dict(ck["state_dict"])
    freeze(net)
    return net


def freeze(net: nn.Module) -> nn.Module:
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# =============================================================================
# COUNT LOSS
# =============================================================================

def sum_pool(density: torch.Tensor, grid: int) -> torch.Tensor:
    """(B, 1, H, W) density -> (B, grid, grid) counts per cell."""
    b, _, h, w = density.shape
    if h % grid or w % grid:
        raise ValueError(f"COUNT_GRID_SIZE={grid} must divide the map size {h}x{w}.")
    return density.reshape(b, grid, h // grid, grid, w // grid).sum(dim=(2, 4))


def count_loss_from_maps(M_f: torch.Tensor, M_r: torch.Tensor, confidence_weight: torch.Tensor,
                         grid: int, q95: float, return_parts: bool = False, total_weight: float = 0.5):
    """Brief step 3, exactly:
      e = (n_f - n_r) / sqrt(n_r + 1)          per cell, Pearson-standardised
      rho = clip(n_r / Q95, 0, 1)
      w_over = 3(1 - rho) + rho ;  w_under = (1 - rho) + 3 rho
      L_cell = mean_cells smoothL1(e) * (w_over if e > 0 else w_under)
      E = (N_f - N_r) / sqrt(N_r + 1) ;  L_total = smoothL1(E)
      L = mean_batch conf * (L_cell + 0.5 L_total)
    M_r must already be detached. Everything in fp32."""
    M_f = M_f.float(); M_r = M_r.float()
    n_f = sum_pool(M_f, grid); n_r = sum_pool(M_r, grid)
    e = (n_f - n_r) / torch.sqrt(n_r + 1.0)
    rho = torch.clamp(n_r / q95, 0.0, 1.0)
    w_over = 3.0 * (1.0 - rho) + 1.0 * rho
    w_under = 1.0 * (1.0 - rho) + 3.0 * rho
    w = torch.where(e > 0, w_over, w_under)
    hub = F.smooth_l1_loss(e, torch.zeros_like(e), beta=1.0, reduction="none")
    L_cell = (hub * w).mean(dim=(1, 2))
    N_f = n_f.sum(dim=(1, 2)); N_r = n_r.sum(dim=(1, 2))
    E = (N_f - N_r) / torch.sqrt(N_r + 1.0)
    L_total = F.smooth_l1_loss(E, torch.zeros_like(E), beta=1.0, reduction="none")
    per_sample = L_cell + total_weight * L_total   # brief default 0.5; 0 = local-only (A5, D-15)
    L = (per_sample * confidence_weight.float()).mean()
    if return_parts:
        return L, dict(L_cell=L_cell.detach(), L_total=L_total.detach(), N_f=N_f.detach(), N_r=N_r.detach())
    return L


def count_loss(counter: nn.Module, fake_H: torch.Tensor, real_H: torch.Tensor, confidence_weight: torch.Tensor,
               grid: int, q95: float, return_parts: bool = False, total_weight: float = 0.5):
    """Frozen counter, fp32 outside autocast, gradient to fake_H only."""
    dev_type = fake_H.device.type
    with torch.autocast(device_type=dev_type, enabled=False):
        M_f = counter(fake_H[:, 0:1].float())
        with torch.no_grad():
            M_r = counter(real_H[:, 0:1].float())
        return count_loss_from_maps(M_f, M_r.detach(), confidence_weight, grid, q95, return_parts, total_weight)


def lambda_count_schedule(epoch: int, lam: float, start: int = 5, full: int = 10) -> float:
    """0 for epochs < start, linear ramp to lam at epoch `full`, then lam."""
    if epoch < start:
        return 0.0
    if epoch >= full:
        return float(lam)
    return float(lam) * (epoch - start + 1) / (full - start + 1)


# =============================================================================
# DENSITY-STRATIFIED SAMPLER
# =============================================================================

def density_sampler_weights(n_real: np.ndarray, cap: float = 3.0, n_iter: int = 50) -> np.ndarray:
    """Per-patch weights equalising low / med / high buckets (by real
    StarDist count), with any patch's oversampling relative to uniform
    capped at `cap`. Returns probabilities summing to 1."""
    n_real = np.asarray(n_real, float)
    b = cl.density_bucket(n_real)
    N = len(n_real)
    present = [k for k in cl.BUCKET_NAMES if (b == k).any()]
    p = np.zeros(N)
    for k in present:
        p[b == k] = 1.0 / (len(present) * (b == k).sum())
    for _ in range(n_iter):
        p = np.minimum(p, cap / N)
        p = p / p.sum()
        if (p <= cap / N + 1e-12).all():
            break
    return p


def build_density_sampler(n_real: np.ndarray, num_samples: int, cap: float = 3.0, generator=None):
    p = density_sampler_weights(n_real, cap)
    return WeightedRandomSampler(torch.as_tensor(p, dtype=torch.double), num_samples=num_samples,
                                 replacement=True, generator=generator), p
