"""Furniture fidelity: is the real catalog product in the image, and does it still look like itself?

Per (image, catalog item):
  detected     an open-vocabulary detector (OWLv2) finds an object of the item's category
  det_score    detector confidence of the chosen box
  dino_sim     DINOv2 cosine similarity, crop vs product photo (DreamBench "DINO" subject-fidelity score)
  clip_sim     CLIP image-image cosine similarity (DreamBench "CLIP-I")
  identity_ok  the crop is closer (DINOv2) to the right product than to every other catalog product of the same
               category. Interpretable as top-1 retrieval. None when the category has only one product.
  delta_e      CIEDE2000 colour distance, product photo vs crop

Among the detector's boxes for the category, we pick the one most similar to the product photo, so two
chairs in one image do not get mixed up.
"""
from __future__ import annotations

from PIL import Image

from . import color, models
from .manifest import CatalogItem, Row


def _query(item: CatalogItem) -> str:
    return f"a photo of a {item.category.replace('_', ' ')}"


def score_image(row: Row, catalog: dict[str, CatalogItem], det_threshold: float = 0.1, top_k: int = 5) -> list[dict]:
    if not row.items:
        return []
    img = Image.open(row.image_path).convert("RGB")
    items = [catalog[i] for i in row.items]
    queries = sorted({_query(it) for it in items})
    dets = models.detect(img, queries, threshold=det_threshold)

    product_imgs = {it.id: Image.open(it.image_path).convert("RGB") for it in catalog.values()}
    out = []
    for it in items:
        q = _query(it)
        cand = [d for d in dets if d["query"] == q][:top_k]
        rec = {"image_id": row.image_id, "item_id": it.id, "category": it.category, "condition": row.condition,
               "detected": bool(cand), "det_score": None, "dino_sim": None, "clip_sim": None,
               "identity_ok": None, "delta_e": None}
        if cand:
            crops = [img.crop(tuple(int(v) for v in d["box"])) for d in cand]
            prod = product_imgs[it.id]
            emb = models.dino_embed([prod] + crops)
            sims = emb[1:] @ emb[0]
            best = int(sims.argmax())
            crop = crops[best]
            rec["det_score"] = cand[best]["score"]
            rec["dino_sim"] = float(sims[best])
            c = models.clip_embed([prod, crop])
            rec["clip_sim"] = float(c[0] @ c[1])
            rec["delta_e"] = color.delta_e(prod, crop)
            rivals = [o for o in catalog.values() if o.category == it.category and o.id != it.id]
            if rivals:
                r_emb = models.dino_embed([product_imgs[o.id] for o in rivals])
                crop_emb = emb[1 + best]
                rec["identity_ok"] = bool(rec["dino_sim"] > float((r_emb @ crop_emb).max()))
        out.append(rec)
    return out
