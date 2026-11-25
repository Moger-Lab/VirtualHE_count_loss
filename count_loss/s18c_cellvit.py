"""
s18c -- CellViT on the s18a smoke inputs (vhe-cellvit env; NOT srs-analysis).
Same output contract as s18b: labels on the 512 grid in
<in_dir>\\det\\cellvit_<variant>__<uid>.npz, timing appended to det\\timing.csv.

Model: CellViT-SAM-H-x40 (PanNuke, 0.25 um/px; Zenodo record 15094831) or
CellViT-256-x40 (--variant HIPT). Image resized 512 -> 400 px (0.25 um/px,
divisible by the 16 px token size), normalised as in the checkpoint config,
softmax on the binary and type maps, the package's own HoVer-style
post-processing at magnification 40, labels resized back to 512 (nearest).

Usage: python s18c_cellvit.py --in_dir D:\\vhe_count_work\\s18_metric_screen\\smoke --variant SAM
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SIZE = 400


def load(variant, dev):
    from cellvit.utils.cache_models import cache_cellvit_sam_h, cache_cellvit_256
    from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
    from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
    from cellvit.utils.tools import unflatten_dict
    path = cache_cellvit_sam_h() if variant == "SAM" else cache_cellvit_256()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    conf = unflatten_dict(ck["config"], ".")
    if ck["arch"] == "CellViTSAM":
        m = CellViTSAM(model_path=None, num_nuclei_classes=conf["data"]["num_nuclei_classes"],
                       num_tissue_classes=conf["data"]["num_tissue_classes"], vit_structure=conf["model"]["backbone"],
                       regression_loss=conf["model"].get("regression_loss", False))
    else:
        m = CellViT256(model256_path=None, num_nuclei_classes=conf["data"]["num_nuclei_classes"],
                       num_tissue_classes=conf["data"]["num_tissue_classes"],
                       regression_loss=conf["model"].get("regression_loss", False))
    print(m.load_state_dict(ck["model_state_dict"]))
    m.eval().to(dev)
    norm = conf.get("transformations", {}).get("normalize", {})
    mean = np.array(norm.get("mean", (0.5, 0.5, 0.5)), np.float32)
    std = np.array(norm.get("std", (0.5, 0.5, 0.5)), np.float32)
    print(f"arch {ck['arch']}, normalize mean {mean} std {std}")
    return m, mean, std


def run(m, mean, std, rgb, dev):
    from skimage.transform import resize
    img = resize(rgb, (SIZE, SIZE), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32) / 255.0
    img = (img - mean) / std
    x = torch.from_numpy(img.transpose(2, 0, 1)[None].copy()).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = m(x)
    out = {k: v.float() for k, v in out.items() if k in ("nuclei_binary_map", "hv_map", "nuclei_type_map")}
    out["nuclei_binary_map"] = F.softmax(out["nuclei_binary_map"], dim=1)
    out["nuclei_type_map"] = F.softmax(out["nuclei_type_map"], dim=1)
    inst, _ = m.calculate_instance_map(out, magnification=40)
    lab = inst[0].numpy().astype(np.int32)
    return resize(lab, (512, 512), order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--variant", default="SAM", choices=["SAM", "HIPT"])
    a = ap.parse_args()
    dev = torch.device("cuda")
    dname = f"cellvit_{a.variant.lower()}"
    out_dir = os.path.join(a.in_dir, "det"); os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter(); m, mean, std = load(a.variant, dev); print(f"{dname}: loaded in {time.perf_counter() - t0:.1f} s")
    timing = []; first = True
    for f in sorted(glob.glob(os.path.join(a.in_dir, "*.npz"))):
        uid = os.path.basename(f)[:-4]
        dst = os.path.join(out_dir, f"{dname}__{uid}.npz")
        if os.path.exists(dst):
            continue
        z = np.load(f); store = {}
        for k in [k for k in z.files if (k.endswith("_rgb") or "_rgb_" in k) and not k.startswith("sd_lab")]:
            if first:
                run(m, mean, std, z[k], dev); first = False
            torch.cuda.synchronize(); t = time.perf_counter()
            store[k] = run(m, mean, std, z[k], dev)
            torch.cuda.synchronize()
            timing.append(dict(detector=dname, patch_uid=uid, image=k, sec=time.perf_counter() - t,
                               n=int(len(np.unique(store[k])) - 1)))
        np.savez_compressed(dst, **store)
        print(f"  {uid}: " + " ".join(f"{k}={len(np.unique(v)) - 1}" for k, v in store.items()), flush=True)
    tp = os.path.join(out_dir, "timing.csv")
    old = pd.read_csv(tp) if os.path.exists(tp) else pd.DataFrame()
    pd.concat([old, pd.DataFrame(timing)]).to_csv(tp, index=False)
    print(pd.DataFrame(timing).sec.describe())
