"""Input contract: one manifest row per generated image, plus the furniture catalog.

manifest.csv columns
    image_id     unique id of the generated image
    image_path   path to the generated image (relative to the manifest file)
    buyer_id     the buyer the image was generated FOR
    condition    personalized | generic | swapped
    group_id     images that share room + brief (A/B pairs are only made inside a group)
    room_image   the empty input room the image was conditioned on (optional, for geometry)
    items        catalog ids that should appear in the image, separated by ";" (optional)

catalog.json is the same shape as data/fixtures/furniture.json in the group repo:
    [{"id": ..., "name": ..., "category": ..., "image_path": ...}, ...]
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

CONDITIONS = {"personalized", "generic", "swapped"}


@dataclass
class Row:
    image_id: str
    image_path: Path
    buyer_id: str
    condition: str
    group_id: str
    room_image: Path | None = None
    items: list[str] = field(default_factory=list)


@dataclass
class CatalogItem:
    id: str
    name: str
    category: str
    image_path: Path


def _resolve(base: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(path: str | Path) -> list[Row]:
    path = Path(path)
    base = path.parent
    rows: list[Row] = []
    seen: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            cond = r["condition"].strip().lower()
            if cond not in CONDITIONS:
                raise ValueError(f"line {i}: condition '{cond}' must be one of {sorted(CONDITIONS)}")
            iid = r["image_id"].strip()
            if iid in seen:
                raise ValueError(f"line {i}: duplicate image_id '{iid}'")
            seen.add(iid)
            room = (r.get("room_image") or "").strip()
            items = [x.strip() for x in (r.get("items") or "").split(";") if x.strip()]
            rows.append(Row(
                image_id=iid,
                image_path=_resolve(base, r["image_path"].strip()),
                buyer_id=r["buyer_id"].strip(),
                condition=cond,
                group_id=r["group_id"].strip(),
                room_image=_resolve(base, room) if room else None,
                items=items,
            ))
    return rows


def load_catalog(path: str | Path) -> dict[str, CatalogItem]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for d in data:
        img = str(d["image_path"]).lstrip("/") if not Path(d["image_path"]).exists() else d["image_path"]
        out[d["id"]] = CatalogItem(
            id=d["id"], name=d.get("name", d["id"]), category=d["category"],
            image_path=_resolve(path.parent, img),
        )
    return out


def check(rows: list[Row], catalog: dict[str, CatalogItem] | None) -> list[str]:
    """Return a list of problems (missing files, unknown catalog ids). Empty list = ok."""
    problems = []
    for r in rows:
        if not r.image_path.exists():
            problems.append(f"{r.image_id}: image not found: {r.image_path}")
        if r.room_image and not r.room_image.exists():
            problems.append(f"{r.image_id}: room image not found: {r.room_image}")
        for it in r.items:
            if catalog is None:
                problems.append(f"{r.image_id}: lists items but no catalog given")
                break
            if it not in catalog:
                problems.append(f"{r.image_id}: unknown catalog id '{it}'")
            elif not catalog[it].image_path.exists():
                problems.append(f"catalog {it}: product image not found: {catalog[it].image_path}")
    return problems
