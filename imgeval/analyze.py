"""Turn the automatic scores and the human answers into one report (markdown + json)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import stats
from .table import flag, group, mean, median, num, read


def _f(x, spec: str) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def _fmt_rate(r: dict) -> str:
    if not r.get("n"):
        return "no data"
    s = f"{r['rate']:.0%} ({r['k']}/{r['n']}), 95% CI {r['wilson_lo']:.0%}–{r['wilson_hi']:.0%}"
    if "boot_lo" in r:
        s += f", rater-bootstrap CI {r['boot_lo']:.0%}–{r['boot_hi']:.0%}"
    if "p_binom" in r:
        s += f", p = {r['p_binom']:.3f}"
    if "per_rater" in r:
        s += f"; {r['raters_above_p0']}/{len(r['per_rater'])} raters above 50%"
    return s


def _fmt_mean(m: dict, nd: int = 2) -> str:
    if not m.get("n"):
        return "no data"
    return f"{m['mean']:.{nd}f} (95% CI {m['lo']:.{nd}f}–{m['hi']:.{nd}f}, n = {m['n']})"


def load_answers(study: Path, answers_dir: Path) -> list[dict]:
    key = json.loads((study / "key.json").read_text("utf-8"))
    files = sorted(Path(answers_dir).glob("answers_*.csv"))
    if not files:
        raise FileNotFoundError(f"no answers_*.csv in {answers_dir}")
    out = []
    for p in files:
        for a in read(p):
            k = key.get(a["trial_id"])
            if k is None:
                continue  # answers from another study build
            out.append({**{kk: v for kk, v in k.items() if kk not in ("rater", "type")}, **a})
    return out


def run(study: Path | None, answers_dir: Path | None, scores_dir: Path | None, out: Path) -> dict:
    res: dict = {"auto": {}, "human": {}, "agreement": {}}
    lines = ["# Image evaluation report", ""]
    images = items = style = None

    # ---------- automatic ----------
    if scores_dir:
        sd = Path(scores_dir)
        images = read(sd / "images.csv") if (sd / "images.csv").exists() else None
        items = read(sd / "items.csv") if (sd / "items.csv").exists() else None
        style = read(sd / "style.csv") if (sd / "style.csv").exists() else None
        lines += ["## Automatic metrics (per condition)", ""]
        if items:
            lines += ["### Furniture fidelity", "", "| condition | detected | identity top-1 | DINOv2 sim | CLIP-I sim | ΔE2000 (median) |",
                      "|---|---|---|---|---|---|"]
            for c, g in group(items, "condition").items():
                idok = [flag(r["identity_ok"]) for r in g if flag(r["identity_ok"]) is not None]
                row = {"n": len(g), "detected": mean([float(bool(flag(r["detected"]))) for r in g]),
                       "identity": mean([float(x) for x in idok]),
                       "dino": mean([num(r["dino_sim"]) for r in g]), "clip": mean([num(r["clip_sim"]) for r in g]),
                       "delta_e": median([num(r["delta_e"]) for r in g])}
                res["auto"].setdefault("fidelity", {})[c] = row
                lines.append(f"| {c} | {_f(row['detected'], '.0%')} | {_f(row['identity'], '.0%')} | {_f(row['dino'], '.3f')} | "
                             f"{_f(row['clip'], '.3f')} | {_f(row['delta_e'], '.1f')} |")
            lines.append("")
        if images:
            lines += ["### Quality and geometry", "", "| condition | MUSIQ | CLIP-IQA | depth corr. vs empty room |", "|---|---|---|---|"]
            for c, g in group(images, "condition").items():
                row = {k: mean([num(r.get(k)) for r in g]) for k in ("musiq", "clipiqa", "depth_corr")}
                res["auto"].setdefault("quality", {})[c] = row
                lines.append(f"| {c} | {_f(row['musiq'], '.1f')} | {_f(row['clipiqa'], '.3f')} | {_f(row['depth_corr'], '.2f')} |")
            lines.append("")
        if (sd / "set.json").exists():
            s = json.loads((sd / "set.json").read_text("utf-8"))
            res["auto"]["set"] = s
            if s.get("kid_x1000") is not None:
                lines += [f"KID ×1000 vs real interior photos: {s['kid_x1000']:.1f} ± {s['kid_std_x1000']:.1f} "
                          f"(n_gen = {s['n_gen']}, n_real = {s['n_real']}). Lower = closer to real photos.", ""]
            elif s.get("note"):
                lines += [f"KID not computed: {s['note']}.", ""]
        if style:
            lines += ["### Style fit (taste-model proxy, partly circular)", "", "| condition | mean score under own buyer's taste |", "|---|---|"]
            for c, g in group(style, "condition").items():
                m = mean([num(r["own_score"]) for r in g])
                res["auto"].setdefault("style", {})[c] = m
                lines.append(f"| {c} | {_f(m, '.4f')} |")
            ranks = [num(r["id_rank"]) for r in style if r["condition"] == "personalized" and r.get("id_rank")]
            if ranks:
                n_b = len([c for c in style[0] if c.startswith("score_")])
                top1 = mean([float(x == 1) for x in ranks])
                res["auto"]["style"]["id_top1"] = top1
                lines += ["", f"Taste identification: for {top1:.0%} of personalized images, the intended buyer's taste scores "
                          f"highest of {n_b} buyers (chance = {1 / max(n_b, 1):.0%})."]
            lines.append("")

    # ---------- human ----------
    rt = pr = []
    if study and answers_dir:
        a = load_answers(Path(study), Path(answers_dir))
        raters = sorted({r["rater"] for r in a})
        res["human"].update(n_raters=len(raters), n_answers=len(a))
        lines += ["## Human study", "", f"{len(raters)} raters, {len(a)} answers.", ""]

        ab = [r for r in a if r["type"] == "ab"]
        if ab:
            lines += ["### H1 Style fit: personalized image preferred by its own buyer (blind 2AFC)", ""]
            for opp, g in group(ab, "opponent_condition").items():
                r = stats.rate([x["choice"] == x["personalized_side"] for x in g], [x["rater"] for x in g])
                res["human"].setdefault("ab", {})[opp] = r
                lines.append(f"- vs **{opp}**: {_fmt_rate(r)}")
            lines.append("")

        rt = [r for r in a if r["type"] == "rate"]
        if rt:
            lines += ["### H3 Quality (MOS 1–5) and realism", ""]
            gen = [r for r in rt if r["source"] == "generated"]
            for src, g in group(rt, "source").items():
                m = stats.mean_ci([num(x["quality"]) for x in g], [x["rater"] for x in g])
                res["human"].setdefault("mos", {})[src] = m
                lines.append(f"- MOS **{src}**: {_fmt_mean(m)}")
                if src == "generated":
                    for c, gc in group(gen, "condition").items():
                        mc = stats.mean_ci([num(x["quality"]) for x in gc], [x["rater"] for x in gc])
                        res["human"].setdefault("mos_by_condition", {})[c] = mc
                        lines.append(f"  - {c}: {_fmt_mean(mc)}")
            if gen:
                fool = stats.rate([x["real"] == "Real photo" for x in gen], [x["rater"] for x in gen], alternative="two-sided")
                fool.pop("p_binom", None)
                res["human"]["fooling_rate"] = fool
                lines.append(f"- Fooling rate (AI image judged real): {_fmt_rate(fool)}")
            real = [r for r in rt if r["source"] == "real"]
            if real:
                base = stats.rate([x["real"] == "Real photo" for x in real], [x["rater"] for x in real], alternative="two-sided")
                base.pop("p_binom", None)
                res["human"]["real_judged_real"] = base
                lines.append(f"- Baseline (real photo judged real): {_fmt_rate(base)}. A fooling rate close to this means "
                             "raters can't tell generated from real (HYPE-style reading).")
            lines.append("")

        pr = [r for r in a if r["type"] == "product"]
        if pr:
            same = [x["product"] == "Yes, same product" for x in pr]
            r = stats.rate(same, [x["rater"] for x in pr], alternative="two-sided")
            r.pop("p_binom", None)
            counts: dict[str, int] = {}
            for x in pr:
                counts[x["product"]] = counts.get(x["product"], 0) + 1
            res["human"].update(product_same=r, product_counts=counts)
            lines += ["### H2 Furniture fidelity (human)", "", f"- Judged the exact same product: {_fmt_rate(r)}",
                      "- All answers: " + ", ".join(f"{k}: {v}" for k, v in counts.items()), ""]

        # ---------- agreement: can we trust the cheap metrics? ----------
        lines += ["## Do the automatic metrics agree with people?", "",
                  "Spearman ρ per image, automatic score vs the human mean. ρ ≥ 0.5 = usable as a proxy on this data.", ""]
        if images and rt:
            mos: dict[str, list[float]] = {}
            for x in rt:
                if x["source"] == "generated":
                    mos.setdefault(x["image_id"], []).append(num(x["quality"]))
            for m in ("musiq", "clipiqa"):
                pairs = [(num(r.get(m)), mean(mos[r["image_id"]])) for r in images if r["image_id"] in mos]
                ag = stats.agreement([p[0] for p in pairs], [p[1] for p in pairs])
                res["agreement"][f"{m}_vs_mos"] = ag
                lines.append(f"- {m} vs MOS: ρ = {_f(ag['rho'], '.2f')} (n = {ag['n']})")
        if items and pr:
            hp: dict[tuple, list[float]] = {}
            for x in pr:
                hp.setdefault((x["image_id"], x["item_id"]), []).append(float(x["product"] == "Yes, same product"))
            pairs = [(num(r["dino_sim"]), mean(hp[(r["image_id"], r["item_id"])])) for r in items
                     if (r["image_id"], r["item_id"]) in hp]
            ag = stats.agreement([p[0] for p in pairs], [p[1] for p in pairs])
            res["agreement"]["dino_vs_human_same"] = ag
            lines.append(f"- DINOv2 sim vs human 'same product': ρ = {_f(ag['rho'], '.2f')} (n = {ag['n']})")
        lines.append("")

    lines += ["## Read with care", "",
              "- The raters are the project group: few, not naive, and they know the system. Treat the results as a pilot.",
              "- Style-fit scores from the taste model are partly circular, because the same model steers generation.",
              "- A detector miss can come from the detector, not the image; spot-check the crops before trusting 'detected'.", ""]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), "utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float), "utf-8")
    return res
