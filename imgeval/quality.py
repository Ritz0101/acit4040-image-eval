"""Image quality and realism, automatic part.

  musiq       MUSIQ no-reference quality (Ke et al., ICCV 2021), trained on KonIQ; higher = better (~0-100)
  clipiqa     CLIP-IQA no-reference quality (Wang et al., AAAI 2023); higher = better (0-1)
  depth_corr  Spearman correlation between the depth map of the empty input room and the generated image.
              Checks the actual apartment's geometry survived (walls, windows, floor). Furniture changes depth
              locally, so 1.0 is not expected; compare conditions with each other, and flag low outliers.
KID (set level, needs a folder of real interior photos): Kernel Inception Distance (Binkowski et al., ICLR 2018).
Unbiased for small sets, unlike FID, which needs thousands of images.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from . import models
from .manifest import Row


@lru_cache(maxsize=None)
def _iqa(name: str):
    import pyiqa
    return pyiqa.create_metric(name, device=models.DEVICE)


def iqa(path: Path) -> dict:
    out = {}
    for name in ("musiq", "clipiqa"):
        try:
            out[name] = float(_iqa(name)(str(path)).item())
        except Exception as e:  # pyiqa missing or weights unavailable: report, don't crash the run
            out[name] = None
            out[f"{name}_error"] = type(e).__name__
    return out


def depth_consistency(room: Path, generated: Path, size: int = 256, n: int = 20000, seed: int = 0) -> float:
    a = models.depth(Image.open(room))
    b = models.depth(Image.open(generated))
    a = np.asarray(Image.fromarray(a).resize((size, size)))
    b = np.asarray(Image.fromarray(b).resize((size, size)))
    idx = np.random.default_rng(seed).choice(size * size, size=min(n, size * size), replace=False)
    return float(spearmanr(a.ravel()[idx], b.ravel()[idx]).statistic)


def score_image(row: Row) -> dict:
    rec = {"image_id": row.image_id, "condition": row.condition, **iqa(row.image_path)}
    rec["depth_corr"] = depth_consistency(row.room_image, row.image_path) if row.room_image else None
    return rec


def kid(generated: list[Path], real_dir: Path, subset: int = 50) -> dict:
    """KID x 1000 (the usual reporting scale) with its std over subsets. Needs >= ~50 images per side."""
    import torch
    from torchmetrics.image.kid import KernelInceptionDistance

    real = sorted(p for p in Path(real_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    k = min(subset, len(generated), len(real))
    if k < 10:
        return {"kid_x1000": None, "kid_std_x1000": None, "note": f"too few images (gen={len(generated)}, real={len(real)})"}
    metric = KernelInceptionDistance(subset_size=k, normalize=True).to(models.DEVICE)

    def batch(paths):
        arr = [np.asarray(Image.open(p).convert("RGB").resize((299, 299)), dtype=np.float32) / 255 for p in paths]
        return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).to(models.DEVICE)

    for i in range(0, len(real), 32):
        metric.update(batch(real[i:i + 32]), real=True)
    for i in range(0, len(generated), 32):
        metric.update(batch(generated[i:i + 32]), real=False)
    m, s = metric.compute()
    return {"kid_x1000": float(m) * 1000, "kid_std_x1000": float(s) * 1000, "n_gen": len(generated), "n_real": len(real)}
