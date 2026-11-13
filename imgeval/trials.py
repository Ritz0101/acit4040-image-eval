"""Build the blind human study from the manifest.

Writes a self-contained study folder:
  study/index.html              the rating page (copied from rating/index.html)
  study/img/<random>.jpg        every image under a random name, so file names leak nothing
  study/trials_<rater>.json     the trials one rater sees (no conditions in it)
  study/key.json                trial id -> what was really shown (kept for the analysis; don't send it to raters)

Three trial types:
  ab       (style fit)   rater = buyer. Their personalized image vs a generic or swapped image from the same group.
                         "Which room fits your taste better?" Left/right randomised.
  rate     (quality + realism) one image, generated or a real interior photo mixed in.
                         Quality 1-5 (ITU-T P.910 ACR scale) and "real photo or AI-generated?".
  product  (furniture fidelity) catalog photo + generated image. "Is this exact product in the room?"
"""
from __future__ import annotations

import json
import random
import secrets
import shutil
from pathlib import Path

from .manifest import CatalogItem, Row

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def build(rows: list[Row], catalog: dict[str, CatalogItem] | None, raters: list[str], out: Path,
          real_dir: Path | None = None, rate_per_rater: int = 30, product_per_rater: int = 10, seed: int = 0) -> dict:
    rng = random.Random(seed)
    out = Path(out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    names: dict[Path, str] = {}

    def blind(p: Path) -> str:
        if p not in names:
            names[p] = f"img/{secrets.token_hex(6)}{p.suffix.lower()}"
            shutil.copy2(p, out / names[p])
        return names[p]

    key: dict[str, dict] = {}
    per_rater: dict[str, list[dict]] = {r: [] for r in raters}

    def add(rater: str, trial: dict, truth: dict):
        tid = f"t{len(key):04d}"
        trial["id"] = tid
        key[tid] = {"rater": rater, "type": trial["type"], **truth}
        per_rater[rater].append(trial)

    by_group: dict[str, list[Row]] = {}
    for r in rows:
        by_group.setdefault(r.group_id, []).append(r)

    # ab: style fit, judged by the buyer the image was made for
    for rater in raters:
        for g, members in by_group.items():
            mine = [r for r in members if r.condition == "personalized" and r.buyer_id == rater]
            rivals = [r for r in members if r.condition == "generic"
                      or (r.condition == "swapped" and r.buyer_id == rater)]
            for p in mine:
                for o in rivals:
                    left_is_p = rng.random() < 0.5
                    a, b = (p, o) if left_is_p else (o, p)
                    add(rater, {"type": "ab", "images": [blind(a.image_path), blind(b.image_path)]},
                        {"personalized_side": "A" if left_is_p else "B", "personalized": p.image_id,
                         "opponent": o.image_id, "opponent_condition": o.condition, "group_id": g})

    # rate: quality + real-or-AI, generated images mixed with real photos (~1 real per 3 generated)
    real = sorted(p for p in Path(real_dir).iterdir() if p.suffix.lower() in IMG_EXT) if real_dir else []
    for rater in raters:
        gen = rng.sample(rows, min(rate_per_rater, len(rows)))
        reals = rng.sample(real, min(len(real), max(1, len(gen) // 3))) if real else []
        pool = [("gen", r) for r in gen] + [("real", p) for p in reals]
        rng.shuffle(pool)
        for kind, x in pool:
            if kind == "gen":
                add(rater, {"type": "rate", "images": [blind(x.image_path)]},
                    {"source": "generated", "image_id": x.image_id, "condition": x.condition})
            else:
                add(rater, {"type": "rate", "images": [blind(x)]}, {"source": "real", "image_id": x.name})

    # product: is the exact catalog item in the image?
    if catalog:
        pairs = [(r, it) for r in rows for it in r.items if it in catalog]
        for rater in raters:
            for r, it in rng.sample(pairs, min(product_per_rater, len(pairs))):
                add(rater, {"type": "product", "images": [blind(catalog[it].image_path), blind(r.image_path)],
                            "label": catalog[it].category.replace("_", " ")},
                    {"image_id": r.image_id, "item_id": it, "condition": r.condition})

    for rater, trials in per_rater.items():
        # ab first (needs the buyer's fresh attention), then the rest shuffled
        ab = [t for t in trials if t["type"] == "ab"]
        rest = [t for t in trials if t["type"] != "ab"]
        rng.shuffle(ab)
        rng.shuffle(rest)
        (out / f"trials_{rater}.json").write_text(json.dumps({"rater": rater, "trials": ab + rest}, indent=1), "utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=1), "utf-8")
    src = Path(__file__).resolve().parent.parent / "rating" / "index.html"
    shutil.copy2(src, out / "index.html")
    return {r: len(t) for r, t in per_rater.items()}
