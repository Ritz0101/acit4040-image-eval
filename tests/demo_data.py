"""Synthetic demo set: flat-shaded 'rooms' with coloured boxes as furniture.

Only for testing that the harness runs end to end. The scores it produces mean nothing.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BUYERS = ["anna", "bo", "cy"]
PRODUCTS = [
    ("sofa-001", "sofa", (196, 180, 150)),
    ("sofa-002", "sofa", (60, 70, 110)),
    ("chair-001", "chair", (150, 100, 60)),
    ("chair-002", "chair", (40, 40, 40)),
]


def _room(wall, floor, size=(640, 448)) -> Image.Image:
    img = Image.new("RGB", size, wall)
    d = ImageDraw.Draw(img)
    w, h = size
    d.polygon([(0, h), (w, h), (int(w * .8), int(h * .62)), (int(w * .2), int(h * .62))], fill=floor)
    d.rectangle([int(w * .4), int(h * .15), int(w * .6), int(h * .45)], fill=(220, 235, 250))  # window
    return img


def make(root: Path, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    root = Path(root)
    for sub in ("catalog", "gen", "rooms", "real", "tastes"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    cat = []
    for pid, category, col in PRODUCTS:
        img = Image.new("RGB", (256, 256), (255, 255, 255))
        ImageDraw.Draw(img).rectangle([40, 90, 216, 200], fill=col)
        img.save(root / "catalog" / f"{pid}.png")
        cat.append({"id": pid, "name": pid, "category": category, "image_path": f"catalog/{pid}.png"})
    (root / "catalog.json").write_text(json.dumps(cat, indent=1), "utf-8")

    rows = []
    for g in range(2):
        room = _room((235, 230, 220), (170, 140, 110))
        room.save(root / "rooms" / f"room{g}.png")
        for cond, buyers in (("generic", ["anna"]), ("personalized", BUYERS), ("swapped", BUYERS)):
            for b in buyers:
                img = room.copy()
                d = ImageDraw.Draw(img)
                sofa, chair = PRODUCTS[int(rng.integers(0, 2))], PRODUCTS[2 + int(rng.integers(0, 2))]
                d.rectangle([60, 300, 330, 400], fill=sofa[2])
                d.rectangle([420, 310, 520, 410], fill=chair[2])
                iid = f"g{g}-{cond}-{b}"
                img.save(root / "gen" / f"{iid}.png")
                rows.append({"image_id": iid, "image_path": f"gen/{iid}.png", "buyer_id": b, "condition": cond,
                             "group_id": f"g{g}", "room_image": f"rooms/room{g}.png", "items": f"{sofa[0]};{chair[0]}"})
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for i in range(4):
        _room(tuple(int(v) for v in rng.integers(150, 255, 3)), (120, 100, 80)).save(root / "real" / f"real{i}.png")
    for b in BUYERS:
        np.save(root / "tastes" / f"{b}.npy", rng.normal(size=768).astype(np.float32))
    return {"manifest": root / "manifest.csv", "catalog": root / "catalog.json", "n": len(rows)}


if __name__ == "__main__":
    import sys
    print(make(Path(sys.argv[1] if len(sys.argv) > 1 else "demo")))
