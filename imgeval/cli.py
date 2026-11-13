"""Command line.

  python -m imgeval check   --manifest M --catalog C
  python -m imgeval score   --manifest M --catalog C [--tastes DIR] [--real-dir DIR] --out results/
  python -m imgeval study   --manifest M --catalog C --raters p1,p2,p3,p4,p5 [--real-dir DIR] --out study/
  python -m imgeval analyze [--study study/ --answers answers/] [--scores results/] --out report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import check, load_catalog, load_manifest
from .table import write


def _load(args):
    rows = load_manifest(args.manifest)
    catalog = load_catalog(args.catalog) if args.catalog else None
    problems = check(rows, catalog)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit(f"{len(problems)} problem(s) in the input; fix them first.")
    return rows, catalog


def cmd_check(args):
    rows, _ = _load(args)
    by: dict[str, int] = {}
    for r in rows:
        by[r.condition] = by.get(r.condition, 0) + 1
    print(f"ok: {len(rows)} images, {len({r.buyer_id for r in rows})} buyers, {len({r.group_id for r in rows})} groups, {by}")


def cmd_score(args):
    from . import fidelity, quality, style

    rows, catalog = _load(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_recs, item_recs = [], []
    for n, r in enumerate(rows, 1):
        print(f"[{n}/{len(rows)}] {r.image_id}", flush=True)
        img_recs.append(quality.score_image(r))
        if catalog and r.items:
            item_recs.extend(fidelity.score_image(r, catalog, det_threshold=args.det_threshold))
    write(out / "images.csv", img_recs)
    if item_recs:
        write(out / "items.csv", item_recs)
    if args.tastes:
        write(out / "style.csv", style.score(rows, style.load_tastes(args.tastes)))
    set_level = quality.kid([r.image_path for r in rows], Path(args.real_dir)) if args.real_dir else {}
    (out / "set.json").write_text(json.dumps(set_level, indent=1), "utf-8")
    print(f"wrote {out}")


def cmd_study(args):
    from . import trials

    rows, catalog = _load(args)
    counts = trials.build(rows, catalog, [x.strip() for x in args.raters.split(",") if x.strip()], Path(args.out),
                          real_dir=Path(args.real_dir) if args.real_dir else None,
                          rate_per_rater=args.rate_per_rater, product_per_rater=args.product_per_rater, seed=args.seed)
    print(f"study written to {args.out}: trials per rater {counts}")
    print(f"serve it with:  python -m http.server -d {args.out} 8000   then open http://localhost:8000/?rater=<id>")
    print("keep key.json out of what you send to raters.")


def cmd_analyze(args):
    from . import analyze

    analyze.run(Path(args.study) if args.study else None, Path(args.answers) if args.answers else None,
                Path(args.scores) if args.scores else None, Path(args.out))
    print(f"wrote {args.out} and {Path(args.out).with_suffix('.json')}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="imgeval", description="Evaluate generated interior images.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--manifest", required=True)
        sp.add_argument("--catalog")

    sp = sub.add_parser("check"); common(sp); sp.set_defaults(f=cmd_check)
    sp = sub.add_parser("score"); common(sp)
    sp.add_argument("--tastes", help="folder with <buyer_id>.npy taste vectors (768-dim CLIP ViT-L/14)")
    sp.add_argument("--real-dir", help="folder of real interior photos, for KID")
    sp.add_argument("--det-threshold", type=float, default=0.1,
                    help="OWLv2 confidence needed to count a product as found (tune on ~10 hand-checked images)")
    sp.add_argument("--out", required=True); sp.set_defaults(f=cmd_score)
    sp = sub.add_parser("study"); common(sp)
    sp.add_argument("--raters", required=True, help="comma-separated rater ids; must equal buyer_ids for the A/B test")
    sp.add_argument("--real-dir"); sp.add_argument("--out", required=True)
    sp.add_argument("--rate-per-rater", type=int, default=30); sp.add_argument("--product-per-rater", type=int, default=10)
    sp.add_argument("--seed", type=int, default=0); sp.set_defaults(f=cmd_study)
    sp = sub.add_parser("analyze")
    sp.add_argument("--study"); sp.add_argument("--answers"); sp.add_argument("--scores")
    sp.add_argument("--out", required=True); sp.set_defaults(f=cmd_analyze)

    args = p.parse_args(argv)
    args.f(args)


if __name__ == "__main__":
    main()
