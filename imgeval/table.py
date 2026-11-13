"""Tiny CSV helpers (no pandas: one less heavy dependency, and some locked-down Windows machines block its DLLs)."""
from __future__ import annotations

import csv
import math
from pathlib import Path


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def flag(v) -> bool | None:
    s = str(v).strip().lower()
    return True if s in {"true", "1", "yes"} else False if s in {"false", "0", "no"} else None


def group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r.get(key)), []).append(r)
    return dict(sorted(out.items()))


def mean(vals) -> float:
    v = [x for x in vals if not math.isnan(x)]
    return sum(v) / len(v) if v else math.nan


def median(vals) -> float:
    v = sorted(x for x in vals if not math.isnan(x))
    if not v:
        return math.nan
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2
