"""
s18b -- second detectors on the s18a smoke inputs (vhe-detectors env; NOT
srs-analysis). Reads the npz files written by s18a, runs each detector on
real, round-trip and fake RGB, and writes label images on the 512 grid
(0.1953 um/px) plus per-image timing. Never touches the s18a files.

Detectors and scale handling (patch = 100 um at 512 px = 0.1953 um/px):
  cellpose   Cellpose 4 default model (cpsam_v2, Cellpose-SAM), RGB at 512,
             diameter=None (model default)
  instanseg  InstanSeg brightfield_nuclei, pixel_size=0.1953 (rescaled
             internally to the model's pixel size, labels returned at 512)
  hovernet   TIAToolbox hovernet_fast-pannuke (0.25 mpp): image resized to
             400 px (0.25 um/px), reflect-padded, run full frame, labels
             resized back to 512 with nearest neighbour

Usage: python s18b_detectors.py --in_dir D:\\vhe_count_work\\s18_metric_screen\\smoke --detectors cellpose instanseg hovernet
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch

UMPP = 100.0 / 512


def to512(lab):
    from skimage.transform import resize
    return resize(lab, (512, 512), order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)


class Cellpose:
    def __init__(self):
        from cellpose import models
        self.m = models.CellposeModel(gpu=True)
        self.name_model = getattr(self.m, "pretrained_model", "default")

    def __call__(self, rgb):
        masks, _, _ = self.m.eval(rgb, diameter=None, channel_axis=-1)
        return np.asarray(masks, np.int32)


class InstanSegDet:
    def __init__(self):
        from instanseg import InstanSeg
        self.m = InstanSeg("brightfield_nuclei", verbosity=0)

    def __call__(self, rgb):
        lab, _ = self.m.eval_small_image(rgb, pixel_size=UMPP, target="nuclei")
        lab = lab.squeeze().cpu().numpy().astype(np.int32)
        if lab.shape != (512, 512):
            lab = to512(lab)
        return lab


class HoverNet:
    SIZE = 400  # 0.25 um/px
    PAD = 46    # valid-conv crop of the fast model is 92 px in total

    def __init__(self):
        from tiatoolbox.models.architecture import get_pretrained_model
        self.m, _ = get_pretrained_model("hovernet_fast-pannuke")
        self.m = self.m.to("cuda").eval()

    def __call__(self, rgb):
        from skimage.transform import resize
        img = resize(rgb, (self.SIZE, self.SIZE), order=1, preserve_range=True, anti_aliasing=True)
        tgt = int(np.ceil((self.SIZE + 2 * self.PAD) / 32.0) * 32)  # encoder needs a multiple of 32
        extra = tgt - self.SIZE - 2 * self.PAD
        p0 = self.PAD + extra // 2; p1 = self.PAD + extra - extra // 2
        img = np.pad(img, ((p0, p1), (p0, p1), (0, 0)), mode="reflect")
        out = self.m.infer_batch(self.m, torch.from_numpy(img[None].astype(np.float32)), device="cuda")
        maps = [o[0] for o in out]
        h = maps[0].shape[0]
        c0 = (h - self.SIZE) // 2
        maps = [mp[c0:c0 + self.SIZE, c0:c0 + self.SIZE] for mp in maps]
        if not hasattr(self, "_checked"):
            print(f"  hovernet: input {img.shape[0]}, output {h}, crop offset {c0} (extra {extra})", flush=True)
            self._checked = True
        lab = self.m.postproc(maps)[0]["predictions"]
        return to512(np.asarray(lab))


DETECTORS = {"cellpose": Cellpose, "instanseg": InstanSegDet, "hovernet": HoverNet}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    a = ap.parse_args()
    out_dir = os.path.join(a.in_dir, "det")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(a.in_dir, "*.npz")))
    timing = []
    for dname in a.detectors:
        t0 = time.perf_counter()
        det = DETECTORS[dname]()
        load_s = time.perf_counter() - t0
        print(f"{dname}: loaded in {load_s:.1f} s", flush=True)
        first = True
        for f in files:
            uid = os.path.basename(f)[:-4]
            dst = os.path.join(out_dir, f"{dname}__{uid}.npz")
            if os.path.exists(dst):
                continue
            z = np.load(f)
            store = {}
            for k in [k for k in z.files if k.endswith("_rgb") or "_rgb_" in k]:
                if k.startswith("sd_lab"):
                    continue
                if first:
                    det(z[k]); first = False  # warm-up, not timed
                torch.cuda.synchronize()
                t = time.perf_counter()
                store[k] = det(z[k])
                torch.cuda.synchronize()
                timing.append(dict(detector=dname, patch_uid=uid, image=k, sec=time.perf_counter() - t,
                                   n=int(len(np.unique(store[k])) - 1)))
            np.savez_compressed(dst, **store)
            print(f"  {uid}: " + " ".join(f"{k}={len(np.unique(v)) - 1}" for k, v in store.items()), flush=True)
        del det
        torch.cuda.empty_cache()
    tp = os.path.join(out_dir, "timing.csv")
    old = pd.read_csv(tp) if os.path.exists(tp) else pd.DataFrame()
    pd.concat([old, pd.DataFrame(timing)]).to_csv(tp, index=False)
    print(pd.DataFrame(timing).groupby("detector").sec.describe())
