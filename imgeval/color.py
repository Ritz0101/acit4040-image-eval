"""Colour fidelity: CIEDE2000 distance between the product photo and the item as rendered.

Product photos usually sit on a white/near-white background, so those pixels are masked out.
Rule of thumb (Sharma et al. 2005; common print/QA practice): dE < 2 barely visible,
2-5 visible on close look, > 10 clearly a different colour.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab


def mean_lab(img: Image.Image, mask_white: bool = True, center: float = 1.0) -> np.ndarray:
    """Mean Lab colour of an image. `center` < 1 keeps only the central part (crops have background at the edges)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    if center < 1.0:
        h, w = arr.shape[:2]
        dh, dw = int(h * (1 - center) / 2), int(w * (1 - center) / 2)
        arr = arr[dh:h - dh or None, dw:w - dw or None]
    lab = rgb2lab(arr).reshape(-1, 3)
    if mask_white:
        keep = ~((lab[:, 0] > 92) & (np.abs(lab[:, 1]) < 4) & (np.abs(lab[:, 2]) < 4))
        if keep.sum() > 0.05 * len(lab):  # fall back to all pixels if the product itself is white
            lab = lab[keep]
    return lab.mean(axis=0)


def delta_e(product: Image.Image, crop: Image.Image) -> float:
    a = mean_lab(product, mask_white=True)
    b = mean_lab(crop, mask_white=False, center=0.6)
    return float(deltaE_ciede2000(a[None, :], b[None, :])[0])
