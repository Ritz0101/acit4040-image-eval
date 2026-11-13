"""Style fit, automatic part: score each image against each buyer's taste vector.

Taste vectors come from the RQ1 model (CLIP ViT-L/14 -> PCA 768->32 -> Bradley-Terry / logistic regression).
Put one file per buyer in a folder: <buyer_id>.npy, a 768-dim weight vector in CLIP space.
If you only have the 32-dim weights w32 and the fitted sklearn PCA, export with:
    w768 = pca.components_.T @ w32;  np.save(f"{buyer}.npy", w768)
(the PCA mean only adds a constant per buyer, which cancels in every comparison below).

CAVEAT: the same taste model steers the generation, so a high score here is partly circular.
It shows the pipeline did what the model asked, not that the buyer likes it. The human blind A/B test is
the primary evidence; this is the cheap, repeatable proxy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import models
from .manifest import Row


def load_tastes(folder: str | Path) -> dict[str, np.ndarray]:
    out = {}
    for p in sorted(Path(folder).glob("*.npy")):
        w = np.load(p).astype(np.float32).ravel()
        if w.shape[0] != 768:
            raise ValueError(f"{p.name}: expected a 768-dim CLIP ViT-L/14 vector, got {w.shape[0]}")
        out[p.stem] = w / (np.linalg.norm(w) + 1e-12)
    return out


def score(rows: list[Row], tastes: dict[str, np.ndarray], batch: int = 16) -> list[dict]:
    """Per image: score under its own buyer's taste, plus taste-identification rank.

    own_score     cosine(image embedding, own buyer's taste direction)
    id_rank       1 = the image's own buyer has the highest score among all buyers (only for personalized images)
    """
    rows = [r for r in rows if r.buyer_id in tastes]
    buyers = sorted(tastes)
    W = np.stack([tastes[b] for b in buyers])  # (B, 768)
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        E = models.clip_embed([Image.open(r.image_path) for r in chunk])  # (n, 768), unit norm
        S = E @ W.T
        for r, s in zip(chunk, S):
            own = float(s[buyers.index(r.buyer_id)])
            rank = int((s > own).sum()) + 1
            out.append({"image_id": r.image_id, "buyer_id": r.buyer_id, "condition": r.condition,
                        "group_id": r.group_id, "own_score": own,
                        "id_rank": rank if r.condition == "personalized" else None,
                        **{f"score_{b}": float(v) for b, v in zip(buyers, s)}})
    return out
