"""
Step 4 unit tests (isolated, synthetic H maps + mock counter where needed).

Run:  python test_step4_units.py            (plain asserts, prints PASS/FAIL)

Covers:
  1. count loss sign and asymmetry (rho = 0 vs rho = 1 cells)
  2. no gradient reaches the counter; gradient does reach fake_H
  3. fp32 path works under AMP (autocast fp16 context)
  4. density sampler bucket frequencies match the target, cap respected
  5. per-core normalisation cache is correct (recomputed independently)
  6. multichannel dataloader and model shapes (1- and 3-channel)
  7. flip convention: a loaded v2 patch pair shows tissue in the same place
  8. COUNT_GRID_SIZE 8 and 2 both pool to correct shapes / conserve counts
  9. target heatmap integral equals count, including border nuclei
 10. lambda_count schedule: 0 through epoch 4, linear to full at epoch 10
"""

import os
import sys
import traceback

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import countlib as cl
import countnet as cn

RESULTS = []


def check(name):
    def deco(fn):
        def run():
            try:
                msg = fn()
                RESULTS.append((name, True, msg or ""))
                print(f"PASS  {name}  {msg or ''}")
            except Exception as e:
                RESULTS.append((name, False, repr(e)))
                print(f"FAIL  {name}")
                traceback.print_exc()
        run.__name__ = fn.__name__
        return run
    return deco


class MockCounter(nn.Module):
    """Density = relu(h + 1) * k: monotone in stain, has parameters so the
    gradient-isolation test is meaningful."""
    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x):
        return torch.relu(x + 1.0) * self.k


def maps_with_cell_counts(nr, nf, grid=8, size=512):
    """Build density maps whose per-cell sums equal nr / nf exactly."""
    cell = size // grid
    Mr = torch.zeros(1, 1, size, size); Mf = torch.zeros(1, 1, size, size)
    for i in range(grid):
        for j in range(grid):
            Mr[0, 0, i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = nr[i][j] / cell ** 2
            Mf[0, 0, i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = nf[i][j] / cell ** 2
    return Mf, Mr


@check("1 count loss sign and asymmetry")
def t_asymmetry():
    q95 = 4.0
    g = 8
    conf = torch.ones(1)

    def loss_for(nr_val, nf_val):
        nr = [[nr_val] * g for _ in range(g)]
        nf = [[nf_val] * g for _ in range(g)]
        Mf, Mr = maps_with_cell_counts(nr, nf, g)
        L, p = cn.count_loss_from_maps(Mf, Mr, conf, g, q95, return_parts=True)
        return p["L_cell"].item()

    # rho = 0 (n_r = 0) vs rho = 1 (n_r >= Q95); same standardised error
    # magnitude |e| = 2 in both, over and under.
    over_sparse = loss_for(0.0, 2.0)                       # e = 2/sqrt(1) = 2
    under_sparse_equiv = None
    nr_d = 8.0; d = 2.0 * np.sqrt(nr_d + 1)               # e = +/-2 at n_r = 8
    over_dense = loss_for(nr_d, nr_d + d)
    under_dense = loss_for(nr_d, nr_d - d)
    # sparse under-generation needs n_r > 0: use n_r small with rho small
    nr_s = 0.4  # rho = 0.1
    ds = 2.0 * np.sqrt(nr_s + 1)
    under_sparse_equiv = loss_for(nr_s, max(nr_s - ds, 0.0))
    over_sparse_small = loss_for(nr_s, nr_s + ds)
    assert over_sparse > over_dense * 2.5, (over_sparse, over_dense)     # w_over 3 vs 1
    assert under_dense > over_dense * 2.5, (under_dense, over_dense)     # dense: under costs more
    assert over_sparse_small > under_sparse_equiv, (over_sparse_small, under_sparse_equiv)  # sparse: over costs more
    # sign: zero error -> zero loss
    assert loss_for(3.0, 3.0) < 1e-10
    return (f"over@rho0={over_sparse:.3f} over@rho1={over_dense:.3f} under@rho1={under_dense:.3f} "
            f"over@rho.1={over_sparse_small:.3f} under@rho.1={under_sparse_equiv:.3f}")


@check("2 no gradient reaches the counter; gradient reaches fake_H")
def t_grad_isolation():
    C = cn.freeze(MockCounter())
    fake = (torch.rand(2, 2, 64, 64) * 2 - 1).requires_grad_(True)
    real = torch.rand(2, 2, 64, 64) * 2 - 1
    L = cn.count_loss(C, fake, real, torch.ones(2), grid=8, q95=2.0)
    L.backward()
    assert C.k.grad is None, "counter parameter received a gradient"
    assert fake.grad is not None and fake.grad.abs().sum() > 0, "no gradient to fake_H"
    assert fake.grad[:, 1].abs().sum() == 0, "gradient leaked into the E channel"
    # an unfrozen counter would get a gradient: confirms the test can fail
    C2 = MockCounter()
    fake2 = (torch.rand(2, 2, 64, 64) * 2 - 1).requires_grad_(True)
    cn.count_loss(C2, fake2, real, torch.ones(2), 8, 2.0).backward()
    assert C2.k.grad is not None
    return "counter grad None, fake_H grad on H channel only"


@check("3 fp32 path under AMP")
def t_amp():
    if not torch.cuda.is_available():
        return "SKIPPED (no CUDA)"
    dev = torch.device("cuda")
    C = cn.freeze(cn.UNetCounter(32).to(dev))
    fake = (torch.rand(2, 2, 512, 512, device=dev) * 2 - 1).requires_grad_(True)
    real = torch.rand(2, 2, 512, 512, device=dev) * 2 - 1
    seen = {}
    def hook(m, i, o):
        seen.setdefault("dtype", o.dtype)  # must return None, or it replaces the output
    h = C.out.register_forward_hook(hook)
    with torch.autocast("cuda", dtype=torch.float16):
        x = fake * 1.0
        assert torch.is_autocast_enabled()
        L = cn.count_loss(C, x, real, torch.ones(2, device=dev), 8, 3.0)
    h.remove()
    assert L.dtype == torch.float32, L.dtype
    assert seen["dtype"] == torch.float32, f"counter ran in {seen['dtype']} inside autocast"
    L.backward()
    assert torch.isfinite(fake.grad).all()
    return f"loss dtype {L.dtype}, counter output dtype {seen['dtype']}"


@check("4 sampler bucket frequencies match target, cap respected")
def t_sampler():
    rng = np.random.default_rng(0)
    n = np.concatenate([rng.integers(3, 33, 600), rng.integers(33, 56, 300), rng.integers(56, 200, 100)])
    p = cn.density_sampler_weights(n, cap=3.0)
    b = cl.density_bucket(n)
    N = len(n)
    target = {k: p[b == k].sum() for k in cl.BUCKET_NAMES}
    assert p.max() <= 3.0 / N + 1e-9, p.max() * N
    # 100 high-density patches of 1000: equal share would need 3.33x -> cap binds at 3x
    assert abs(target["high"] - 0.30) < 1e-6, target
    assert abs(target["low"] - target["med"]) < 1e-6, target
    sampler, _ = cn.build_density_sampler(n, num_samples=N, cap=3.0, generator=torch.Generator().manual_seed(0))
    draws = np.concatenate([np.array(list(iter(sampler))) for _ in range(40)])
    freq = {k: float((b[draws] == k).mean()) for k in cl.BUCKET_NAMES}
    for k in cl.BUCKET_NAMES:
        assert abs(freq[k] - target[k]) < 0.01, (k, freq[k], target[k])
    # uncapped case: equal thirds
    n2 = np.concatenate([rng.integers(3, 33, 500), rng.integers(33, 56, 300), rng.integers(56, 200, 200)])
    p2 = cn.density_sampler_weights(n2, cap=3.0); b2 = cl.density_bucket(n2)
    assert all(abs(p2[b2 == k].sum() - 1 / 3) < 1e-9 for k in cl.BUCKET_NAMES)
    return f"target {{{', '.join(f'{k}: {v:.3f}' for k, v in target.items())}}} drawn {{{', '.join(f'{k}: {v:.3f}' for k, v in freq.items())}}}; epoch length unchanged ({N})"


@check("5 per-core normalisation cache correct")
def t_norm_cache():
    tab = cl.per_core_percentile_table()
    core = "D-3"
    srs, _ = cl.core_arrays(core)
    mask = cl.srs_tissue_mask(srs, 4)
    for c in range(4):
        v = np.asarray(srs[c, ::4, ::4])[mask]
        lo, hi = np.percentile(v, cl.PER_CORE_PERCENTILES)
        assert np.isclose(lo, getattr(tab[core], f"ch{c}_lo")) and np.isclose(hi, getattr(tab[core], f"ch{c}_hi")), c
    # applying it: a synthetic patch at exactly (lo, hi) maps to (-1, +1)
    r = tab[core]
    y = cl.normalize_srs(np.full((1, 4, 4), r.ch1_lo, np.float32), core, [1], "per_core")
    assert np.allclose(y, -1.0)
    y = cl.normalize_srs(np.full((1, 4, 4), r.ch1_hi, np.float32), core, [1], "per_core")
    assert np.allclose(y, 1.0)
    # per_core output does NOT depend on patch content elsewhere (unlike per_patch)
    a = np.random.default_rng(0).uniform(0, 2000, (1, 32, 32)).astype(np.float32)
    b = a.copy(); b[0, :4, :4] = 0
    ya = cl.normalize_srs(a, core, [1], "per_core"); yb = cl.normalize_srs(b, core, [1], "per_core")
    assert np.allclose(ya[0, 8:, 8:], yb[0, 8:, 8:])
    pa = cl.normalize_srs(a, core, [1], "per_patch"); pb = cl.normalize_srs(b, core, [1], "per_patch")
    assert not np.allclose(pa[0, 8:, 8:], pb[0, 8:, 8:])
    # type channels use the global row under TYPE_CHANNEL_NORM='global'
    g = tab["__GLOBAL__"]
    y0 = cl.normalize_srs(np.full((1, 2, 2), g.ch0_hi, np.float32), "H-8", [0], "per_core", "global")
    assert np.allclose(y0, 1.0)
    return f"D-3 recomputed matches; type channels use GLOBAL ch0 [{g.ch0_lo:.0f},{g.ch0_hi:.0f}]"


def _rows(n=2, core="D-3"):
    m, _, _ = cl.load_v2_tables()
    r = m[(m.core == core) & (m.role == "eval")].head(n).copy()
    r["confidence_weight"] = 1.0; r["n_real"] = np.nan
    return r


@check("6 multichannel dataloader and model shapes")
def t_shapes():
    calib = cn.v1_common.load_stain_calibration(cl.CALIBRATION_PATH)
    out = []
    for chans in ([1], [1, 0, 3]):
        cfg = dict(SRS_CHANNEL_INDICES=chans, SRS_NORM_MODE="per_core", NETWORK_INPUT_SIZE=512,
                   SRS_CORE_FLIP_HORIZONTAL=False)
        ds = cn.V2PatchDataset(_rows(), cfg, calib)
        s = ds[0]
        assert s["srs"].shape == (len(chans), 512, 512), s["srs"].shape
        assert s["he"].shape == (2, 512, 512)
        assert s["srs"].min() >= -1 and s["srs"].max() <= 1
        nc = len(chans)
        G_S2H = cn.v1_models.ResnetGenerator(nc, 2, ngf=8, n_residual_blocks=1)
        G_H2S = cn.v1_models.ResnetGenerator(2, nc, ngf=8, n_residual_blocks=1)
        D_S = cn.v1_models.PatchGANDiscriminator(nc, ndf=8)
        x = s["srs"][None]
        fh = G_S2H(x); rs = G_H2S(fh); ds_out = D_S(rs)
        assert fh.shape == (1, 2, 512, 512) and rs.shape == (1, nc, 512, 512)
        assert ds_out.shape[1] == 1
        out.append(f"{nc}ch ok")
    # the dataset refuses the v1 flip on the v2 path
    try:
        cn.V2PatchDataset(_rows(), dict(SRS_CHANNEL_INDICES=[1], SRS_NORM_MODE="per_core",
                                         NETWORK_INPUT_SIZE=512, SRS_CORE_FLIP_HORIZONTAL=True), calib)
        raise AssertionError("dataset accepted SRS_CORE_FLIP_HORIZONTAL=True")
    except ValueError:
        out.append("flip=True refused")
    return ", ".join(out)


@check("7 flip convention: loaded patch pair shows tissue in the same place")
def t_flip():
    """Through the actual dataset class: per patch, correlation between CARS
    and H&E optical density at 8x8 blocks, as loaded vs with the H&E
    mirrored (the double-flip failure). Tissue-boundary patches carry the
    signal, so select eval patches with partial tissue."""
    from skimage.transform import resize as sk_resize
    m, _, _ = cl.load_v2_tables()
    rs, rm = [], []
    for core in ("D-3", "B-5", "E-9", "H-2"):
        rows = m[(m.core == core) & (m.role == "eval")]
        for r in rows.itertuples():
            srs, he = cl.v2_patch_native(core, r.y0, r.x0, 441, channels=[2])
            od = -np.log(np.clip(he.astype(np.float32), 1, 255) / 255).sum(-1)
            a = sk_resize(srs[0], (16, 16), anti_aliasing=True); b = sk_resize(od, (16, 16), anti_aliasing=True)
            if a.std() < 1e-6 or b.std() < 1e-6:
                continue
            rs.append(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            rm.append(np.corrcoef(a.ravel(), b[:, ::-1].ravel())[0, 1])
    rs, rm = np.array(rs), np.array(rm)
    assert np.median(rs) > np.median(rm) + 0.1, (np.median(rs), np.median(rm))
    assert (rs > rm).mean() > 0.7, (rs > rm).mean()
    return f"{len(rs)} patches: median r as loaded {np.median(rs):+.3f} vs mirrored {np.median(rm):+.3f}; as-loaded wins {100 * (rs > rm).mean():.0f}%"


@check("8 COUNT_GRID_SIZE 8 and 2 pooling shapes and count conservation")
def t_grid():
    x = torch.rand(3, 1, 512, 512)
    msgs = []
    for g in (8, 2):
        p = cn.sum_pool(x, g)
        assert p.shape == (3, g, g), p.shape
        assert torch.allclose(p.sum(dim=(1, 2)), x.sum(dim=(1, 2, 3)), rtol=1e-5)
        # a single known cell
        y = torch.zeros(1, 1, 512, 512); c = 512 // g
        y[0, 0, c + 1, 2 * c - 1 if g > 2 else 0] = 5.0
        assert cn.sum_pool(y, g)[0, 1, 1 if g > 2 else 0].item() == 5.0
        L = cn.count_loss_from_maps(x, x.flip(-1), torch.ones(3), g, 3.0)
        assert torch.isfinite(L)
        msgs.append(f"g={g} -> {tuple(p.shape)}")
    try:
        cn.sum_pool(x, 7); raise AssertionError("grid 7 accepted")
    except ValueError:
        pass
    return ", ".join(msgs)


@check("9 heatmap integral equals count (incl. border nuclei)")
def t_heatmap():
    cy = np.array([0.0, 5.0, 250.3, 511.0, 300.0]); cx = np.array([0.0, 511.0, 250.7, 256.0, 1.0])
    for s in (3.0, 6.3):
        hm = cl.heatmap_from_centroids(cy, cx, s)
        assert abs(hm.sum() - len(cy)) < 1e-3, hm.sum()
        assert hm.min() >= 0
    hm0 = cl.heatmap_from_centroids(np.array([]), np.array([]), 5.0)
    assert hm0.sum() == 0
    # an interior Gaussian has close to unit mass without renormalisation
    one = cl.heatmap_from_centroids(np.array([256.0]), np.array([256.0]), 6.0)
    assert abs(one.sum() - 1) < 1e-4 and one.argmax() == 256 * 512 + 256
    return "sums exact for sigma 3 and 6.3; empty map zero"


@check("10 lambda_count schedule")
def t_sched():
    v = [cn.lambda_count_schedule(e, 1.0) for e in range(13)]
    assert v[:5] == [0.0] * 5 and v[10] == 1.0 and v[12] == 1.0
    assert all(np.diff(v[4:11]) > 0), v
    return " ".join(f"{x:.2f}" for x in v)


if __name__ == "__main__":
    for t in (t_asymmetry, t_grad_isolation, t_amp, t_sampler, t_norm_cache, t_shapes, t_flip, t_grid, t_heatmap, t_sched):
        t()
    n_ok = sum(ok for _, ok, _ in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} passed")
    os.makedirs(os.path.join(_script_dir, "outputs", "s04_tests"), exist_ok=True)
    pd.DataFrame(RESULTS, columns=["test", "passed", "detail"]).to_csv(
        os.path.join(_script_dir, "outputs", "s04_tests", "unit_tests.csv"), index=False)
    sys.exit(0 if n_ok == len(RESULTS) else 1)
